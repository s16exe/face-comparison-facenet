import cv2
import numpy as np
from keras_facenet import FaceNet
from mtcnn.mtcnn import MTCNN
from numpy import linalg as LA

# Initialize the MTCNN face detector and FaceNet embedder
detector = MTCNN()
embedder = FaceNet()


def detect_and_embed(image_path):
    """Detect faces in an image and return their embeddings."""
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image at {image_path}")
        return []

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB

    # Detect faces in the image
    results = detector.detect_faces(image)

    embeddings = []
    for result in results:
        # Get the bounding box coordinates
        x, y, width, height = result['box']
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = x + width, y + height

        # Extract the face from the image
        face = image[y1:y2, x1:x2]
        face = cv2.resize(face, (160, 160))  # Resize to 160x160 for FaceNet
        face = np.expand_dims(face, axis=0)  # Add batch dimension

        # Get embeddings for the face
        embedding = embedder.embeddings(face)

        # Normalize the embedding for better comparison
        embedding = embedding / np.linalg.norm(embedding)  # L2 Normalization
        embeddings.append(embedding[0])  # Append the normalized embedding

    return embeddings


def cosine_similarity(embedding1, embedding2):
    """Compute the cosine similarity between two embeddings."""
    dot_product = np.dot(embedding1, embedding2)
    norm_a = np.linalg.norm(embedding1)
    norm_b = np.linalg.norm(embedding2)
    return dot_product / (norm_a * norm_b)


def similarity_percentage(embedding1, embedding2):
    """Convert cosine similarity to percentage for easier interpretation."""
    cos_sim = cosine_similarity(embedding1, embedding2)
    print(cos_sim)
    similarity = (cos_sim + 1) / 2 * 100  # Convert cosine similarity to a 0-100% range
    return similarity


def match_faces(image_path1, image_path2, threshold=80):
    """Match faces from two images and return whether they match based on the threshold."""
    embeddings1 = detect_and_embed(image_path1)
    embeddings2 = detect_and_embed(image_path2)

    if embeddings1 and embeddings2:
        # Compare the first detected face from both images
        similarity = similarity_percentage(embeddings1[0], embeddings2[0])
        print(f"Similarity: {similarity:.2f}%")

        # Check if the similarity exceeds the threshold
        if similarity >= threshold:
            print("Faces match!")
        else:
            print("Faces do not match.")
    else:
        print("No faces detected in one or both images.")


# Example usage
image_path1 = r'C:\Users\SubramanyaChar\PycharmProjects\one-to-one-image-matching\Own\05-10-2024\dataset\sa1.jpg'  # Replace with the path to your first image
image_path2 = r'C:\Users\SubramanyaChar\PycharmProjects\one-to-one-image-matching\Own\05-10-2024\dataset\sa2.jpg'  # Replace with the path to your second image

match_faces(image_path1, image_path2, threshold=80)
