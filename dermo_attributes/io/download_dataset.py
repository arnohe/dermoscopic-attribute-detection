import requests
import zipfile
import shutil
import os

from dermo_attributes.io.paths import isic_path, Splits, Datatypes


def download_and_extract(extract_path, url):
    # Send HTTP request to the URL of the file
    response = requests.get(url, stream=True)

    # Check if the request is successful
    if response.status_code == 200:
        # Write the zip temporary file to disk
        with open('../../temp.zip', 'wb') as temp_zip:
            temp_zip.write(response.content)

        # Create folder for extraction
        if not os.path.exists(extract_path):
            os.makedirs(extract_path)

        # Extract all the contents of the zip file in the specified directory
        with zipfile.ZipFile("../../temp.zip") as zip_file:
            for member in zip_file.namelist():
                filename = os.path.basename(member)
                # Skip zip directories
                if not filename:
                    continue

                # Remove zip directory from file path
                source = zip_file.open(member)
                target = open(os.path.join(extract_path, filename), "wb")
                with source, target:
                    shutil.copyfileobj(source, target)

        # Remove the temporary zip file
        os.remove('../../temp.zip')

    else:
        print(f"Failed to download the file. HTTP response code: {response.status_code}")


def download_dataset():
    print("downloading and extracting validation set to " + isic_path(Splits.VALIDATION))
    valid_input = "https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task1-2_Validation_Input.zip"
    valid_seg = "https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task1_Validation_GroundTruth.zip"
    valid_truth = "https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task2_Validation_GroundTruth.zip"
    download_and_extract(isic_path(Splits.VALIDATION, Datatypes.IN), valid_input)
    download_and_extract(isic_path(Splits.VALIDATION, Datatypes.SEG), valid_seg)
    download_and_extract(isic_path(Splits.VALIDATION, Datatypes.GTX), valid_truth)

    print("downloading and extracting test set to " + isic_path(Splits.TEST))
    test_input = "https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task1-2_Test_Input.zip"
    test_seg = "https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task1_Test_GroundTruth.zip"
    test_truth = "https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task2_Test_GroundTruth.zip"
    download_and_extract(isic_path(Splits.TEST, Datatypes.IN), test_input)
    download_and_extract(isic_path(Splits.TEST, Datatypes.SEG), test_seg)
    download_and_extract(isic_path(Splits.TEST, Datatypes.GTX), test_truth)

    print("downloading and extracting training set to " + isic_path(Splits.TRAIN))
    train_input = "https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task1-2_Training_Input.zip"
    train_seg = "https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task1_Training_GroundTruth.zip"
    train_truth = "https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task2_Training_GroundTruth_v3.zip"
    download_and_extract(isic_path(Splits.TRAIN, Datatypes.IN), train_input)
    download_and_extract(isic_path(Splits.TRAIN, Datatypes.SEG), train_seg)
    download_and_extract(isic_path(Splits.TRAIN, Datatypes.GTX), train_truth)


if __name__ == "__main__":
    download_dataset()
