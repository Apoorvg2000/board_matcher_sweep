# Import required Python packages
import os
import pickle
import time

import cv2 as cv
from flask import Flask, jsonify, request
from logger import init_loggers

# Initialize the loggers
info_logger, debug_logger = init_loggers()

# Define the root directory path
root_dir = "pi_capture"

# Define the paths for the descriptors, heights and widths pickle files
descriptor_file = os.path.join(root_dir, "descriptors.pkl")
heights_file = os.path.join(root_dir, "heights.pkl")
widths_file = os.path.join(root_dir, "widths.pkl")


# Method to find the best matching/closest image in the references images to the input image and return the best matched reference image
def get_code(image_path):
    # Get the reference image paths
    references_folder = os.path.join(root_dir, "data/references")
    all_refereces = os.listdir(references_folder)
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]

    # Create list of all reference image paths
    ref_image_files = [
        f
        for f in all_refereces
        if any(f.lower().endswith(ext) for ext in image_extensions)
    ]

    try:
        image_path = os.path.join(root_dir, image_path)
        total_time = 0.0
        max_matches = 0
        best_match = "no match"

        # Iterate over all reference images
        for idx, image_file in enumerate(ref_image_files):
            ref_filename = os.path.join(references_folder, image_file)

            # Call the cuda_orb_match method to get the time taken and number of good matches for each reference image
            time_taken, good_matches = cuda_orb_match(image_path, ref_filename)

            total_time += time_taken

            # If the number of matches for an images exceeds the max_matches, make that reference image as the best match
            if max_matches < len(good_matches):
                max_matches = len(good_matches)
                best_match = ref_filename

        info_logger.info(f"Total time taken for reference image matching: {total_time}")

        # Return the best match
        return best_match
    except Exception as e:
        debug_logger.debug(
            f"{time.ctime()}: An error occurred during reference image matching {e}"
        )
        return best_match


# Method to compare ORB features of the input image and reference image and return the total time taken by the method and number of good matches
def cuda_orb_match(input_path, reference_img_path):
    # Open the descriptors file and store the contents in a descriptors dictionary
    with open(descriptor_file, "rb") as f:
        try:
            desc_data = pickle.load(f)
        except EOFError:
            desc_data = {}

    # Variable to store initial length of descriptors.pkl file
    init_len = len(desc_data)

    ref_image_basename = os.path.basename(reference_img_path)

    # Time to detect and describe
    start_time_detect_describe = time.time()

    # img1: input image
    img1 = cv.imread(input_path, cv.IMREAD_GRAYSCALE)

    # Ensure the images are not empty
    assert img1 is not None, "could not read image 1"

    # Create the CUDA ORB detector
    orb = cv.cuda_ORB.create()

    # Upload images to GPU memory
    gpu_img1 = cv.cuda_GpuMat(img1)

    # Detect and compute keypoints and descriptors
    keypoints1_gpu, descriptors1_gpu = orb.detectAndComputeAsync(gpu_img1, None)

    # Download descriptors from GPU to CPU memory
    descriptors1 = descriptors1_gpu.download()

    # If the descriptors are not in the descriptors dictionary, then calculate them otherwise use the pre-stored descriptors
    if ref_image_basename not in desc_data:
        # img2: reference image
        img2 = cv.imread(reference_img_path, cv.IMREAD_GRAYSCALE)

        # Ensure the images are not empty
        assert img2 is not None, "Could not read the image 2."

        # Create the CUDA ORB detector
        orb = cv.cuda_ORB.create()

        # Upload images to GPU memory
        gpu_img2 = cv.cuda_GpuMat(img2)

        # Detect and compute keypoints and descriptors
    # Calculate the heights and widths of the images
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # Open the heights and widths files and store the contents in dictionaries
    with open(heights_file, "rb") as f:
        try:
            heights_data = pickle.load(f)
        except EOFError:
            heights_data = {}

    with open(widths_file, "rb") as f:
        try:
            widths_data = pickle.load(f)
        except EOFError:
            widths_data = {}

    # Check if the heights and widths of the images are in the dictionaries
    if input_path not in heights_data:
        heights_data[input_path] = h1
    else:
        h1 = heights_data[input_path]

    if input_path not in widths_data:
        widths_data[input_path] = w1
    else:
        w1 = widths_data[input_path]

    if ref_image_basename not in heights_data:
        heights_data[ref_image_basename] = h2
    else:
        h2 = heights_data[ref_image_basename]

    if ref_image_basename not in widths_data:
        widths_data[ref_image_basename] = w2
    else:
        w2 = widths_data[ref_image_basename]

    # Calculate the ratios of the heights and widths of the images
    h_ratio = h1 / h2
    w_ratio = w1 / w2

    # If either ratio is greater than 1.5, return (0, 0)
    if h_ratio > 1.5 or w_ratio > 1.5:
        return (0, 0)
        keypoints2_gpu, descriptors2_gpu = orb.detectAndComputeAsync(gpu_img2, None)

        # Download descriptors from GPU to CPU memory
        descriptors2 = descriptors2_gpu.download()

        # Store the calculated descriptors for future use
        desc_data[ref_image_basename] = descriptors2
    else:
        descriptors2 = desc_data[ref_image_basename]

    # To check if any new descriptors have been added by comparing the initial length and current length of ther descriptors dictionary
    if init_len != len(desc_data):
        with open(descriptor_file, "wb") as f:
            pickle.dump(desc_data, f)

    end_time_detect_describe = time.time()

    time_taken_detect_describe = end_time_detect_describe - start_time_detect_describe

    # Time to match
    start_time_match = time.time()

    # Matching descriptors using BFMatcher (Brute Force Matcher)
    bf = cv.BFMatcher(cv.NORM_HAMMING)
    knn_matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    # Apply ratio test
    ratio_thresh = 0.75
    good_matches = []
    for m, n in knn_matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)

    end_time_match = time.time()

    time_taken_match = end_time_match - start_time_match

    # Return total time taken and number of good matches
    return (time_taken_match + time_taken_detect_describe, good_matches)
    # If any new heights or widths have been added, write the updated dictionaries back to the files
    with open(heights_file, "wb") as f:
        pickle.dump(heights_data, f)

    with open(widths_file, "wb") as f:
        pickle.dump(widths_data, f)
