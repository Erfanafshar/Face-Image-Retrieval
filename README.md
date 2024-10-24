# Face Recognition Retrieval System

## Table of Contents
1. General Description of the System
2. General Description of Main Folders
3. Folder: template
4. Folder: file divider
5. Folder: part1
   - 5.1 Face Detection from Image
   - 5.2 Feature Vector Generation from Face
   - 5.3 Organizing Faces and Feature Vectors
6. Folder: part2
   - 6.1 Required Files
   - 6.2 User Interface
   - 6.3 Server
7. Important Notes

## 1. General Description of the System
The system is a web-based application using HTML and frontend languages. It consists of two main parts: the backend and frontend. The implementation of the backend has been achieved by making specific modifications and additions to the essential JavaScript and CSS for the primary functionality of the system, which is face searching. The backend is implemented in a single Python file. However, there are additional Python scripts that are also used to complete the system, focusing primarily on generating feature vectors for face images.

## 2. General Description of Main Folders
There are four main folders as follows:

1. **template**: This folder contains the initial layout, which is the frontend of the system.
2. **file divider**: This folder includes scripts that automate the process of dividing a large number of images into smaller groups and renaming files. Manually performing such tasks can be time-consuming, especially in a system with many faces.
3. **part1**: In this section, faces present in the images are detected, and feature vectors are generated for them. The organized collections of images and feature vectors are prepared for use in the main system.
4. **part2**: This is the main system, which consists of two main parts: the website's appearance and the server. The appearance part is implemented using HTML, JavaScript, and CSS, while the server is built around a single Python file.

## 3. Folder: template
This folder includes all the files related to the initial template. To interact with the template, you will need the index.html file, which presents the layout and how it works.

## 4. Folder: file divider
The files in this folder aim to speed up processes that can also be done manually, such as renaming files or copying files in an organized manner. These codes are not complex, and detailed explanations of them are not crucial. Some tasks include:
- Transferring all images from various folders to a single folder.
- Dividing images in a folder into multiple folders of fixed sizes (e.g., 10,000 images per folder).
- Renaming image files numerically to create a defined order.

## 5. Folder: part1
The primary goal in this section is to generate feature vectors for face images. This process involves three steps, each corresponding to a different file:
1. **Face Detection from Image**: This file reads images from the input folder and uses the MTCNN library to detect faces, storing the results in a large array.
2. **Feature Vector Generation from Face**: This section consists of three files that generate feature vectors for the detected faces using specific models. The output from the previous section is also utilized here.
3. **Organizing Faces and Feature Vectors**: This part concatenates the outputs from the previous step, as the system requires images in batches of 100,000, while the previous outputs were in batches of 10,000.

## 6. Folder: part2
This section is the main system, including essential components:
1. **Required Files**: These files are divided into user interface files and server files. The server files include feature vectors for three models and images. User interface files contain images stored in specific folders.
2. **User Interface**: The user interface is implemented using HTML, JavaScript, and CSS. The main site page has been modified, with all relevant files located in the corresponding folders.
3. **Server**: This part has a single Python file named main.py. Before running the code, specific variables need to be initialized based on the current dataset and feature vector collection. It may take up to 15 minutes for the server to start, especially when handling a large number of images. After the server is up, you can access it via http://localhost:8080 in your browser.

## 7. Important Notes
- Files related to the libraries used have been removed to reduce size. Before executing the code, these libraries need to be installed. Given the number of libraries, listing them all for installation isn't practical. The best approach is to run the code and address missing libraries as errors occur.
- The names and versions of the libraries used in the main server code are visible in the accompanying images.
- The system has been implemented and tested in PyCharm version 2021.1.1.
- All Python code files are independent, meaning none require a specific starting point to run. The similarity among files is due to their shared goals.
