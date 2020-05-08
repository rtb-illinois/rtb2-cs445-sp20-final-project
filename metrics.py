import math
import functools
import numpy as np
from scipy.spatial import Delaunay
from morphing import *

def get_max_angle(indeces, landmarks, *args, **kwargs):
    v1 = landmarks[indeces[0], :]
    v2 = landmarks[indeces[1], :]
    v3 = landmarks[indeces[2], :]
    
    s1 = math.sqrt((v1[0] - v2[0])**2 + (v1[1] - v2[1])**2)
    s2 = math.sqrt((v1[0] - v3[0])**2 + (v1[1] - v3[1])**2)
    s3 = math.sqrt((v2[0] - v3[0])**2 + (v2[1] - v3[1])**2)
    
    a1 = math.acos((s2**2 + s3**2 - s1**2) / (2 * s2 * s3));  
    a2 = math.acos((s1**2 + s3**2 - s2**2) / (2 * s1 * s3));  
    a3 = math.acos((s1**2 + s2**2 - s3**2) / (2 * s1 * s2));
    
    return max(a1, a2, a3)


def get_min_angle(indeces, landmarks, *args, **kwargs):
    v1 = landmarks[indeces[0], :]
    v2 = landmarks[indeces[1], :]
    v3 = landmarks[indeces[2], :]
    
    s1 = math.sqrt((v1[0] - v2[0])**2 + (v1[1] - v2[1])**2)
    s2 = math.sqrt((v1[0] - v3[0])**2 + (v1[1] - v3[1])**2)
    s3 = math.sqrt((v2[0] - v3[0])**2 + (v2[1] - v3[1])**2)
    
    a1 = math.acos((s2**2 + s3**2 - s1**2) / (2 * s2 * s3));  
    a2 = math.acos((s1**2 + s3**2 - s2**2) / (2 * s1 * s3));  
    a3 = math.acos((s1**2 + s2**2 - s3**2) / (2 * s1 * s2));
    
    return min(a1, a2, a3)


def get_area_percentage(indeces, landmarks, face_area):
    mask = get_image_mask(900, 1200, np.take(landmarks, indeces, axis=0).astype('int32'))
    area = np.sum(mask, axis=(0,1))[0]
    return area / face_area


def get_face_similarity(face1, face2, similarity_func):
    face1_landmarks = get_face_landmarks(face1)
    face2_landmarks = get_face_landmarks(face2)
    
    average_landmarks = np.average([face1_landmarks, face2_landmarks], axis=0, weights=[0.5, 0.5])

    triangulation = Delaunay(average_landmarks)
    
    triangles = [list(triangle) for triangle in triangulation.simplices]
    
    face_only_triangles = [triangle for triangle in triangles if not (set(triangle) - set(range(68)))]
    
    face1_partial = functools.partial(similarity_func, landmarks=face1_landmarks)
    face1_similarity = list(map(face1_partial, face_only_triangles))

    face2_partial = functools.partial(similarity_func, landmarks=face2_landmarks)
    face2_similarity = list(map(face2_partial, face_only_triangles))
    
    return sum(abs(np.array(face1_similarity) - np.array(face2_similarity)))


def get_face_similarity_test(face1, face2, similarity_funcs):
    face1_landmarks = get_face_landmarks(face1)
    face2_landmarks = get_face_landmarks(face2)
    
    average_landmarks = np.average([face1_landmarks, face2_landmarks], axis=0, weights=[0.5, 0.5])

    triangulation = Delaunay(average_landmarks)
    
    triangles = [list(triangle) for triangle in triangulation.simplices]
    
    face_only_triangles = [triangle for triangle in triangles if not (set(triangle) - set(range(68)))]

    face1_masks = [get_image_mask(900, 1200, np.take(face1_landmarks, triangle, axis=0).astype('int32'))
                   for triangle in face_only_triangles]
    face1_area = np.sum(np.sum(np.array(face1_masks), axis=(1,2))[:, 0])
    face2_masks = [get_image_mask(900, 1200, np.take(face2_landmarks, triangle, axis=0).astype('int32'))
                   for triangle in face_only_triangles]
    face2_area = np.sum(np.sum(np.array(face2_masks), axis=(1,2))[:, 0])
    
    similarities = []
    for similarity_func in similarity_funcs:
        face1_partial = functools.partial(similarity_func, landmarks=face1_landmarks, face_area=face1_area)
        face1_similarity = list(map(face1_partial, face_only_triangles))

        face2_partial = functools.partial(similarity_func, landmarks=face2_landmarks, face_area=face2_area)
        face2_similarity = list(map(face2_partial, face_only_triangles))

        similarities.append(np.mean(abs(np.array(face1_similarity) - np.array(face2_similarity)) / np.array(face2_similarity)))
    return np.mean(similarities)


def get_face_similarity_eval(face1, face2, similarity_func):
    face1_landmarks = get_face_landmarks(face1)
    face2_landmarks = get_face_landmarks(face2)
    
    average_landmarks = np.average([face1_landmarks, face2_landmarks], axis=0, weights=[0.5, 0.5])

    triangulation = Delaunay(average_landmarks)
    
    triangles = [list(triangle) for triangle in triangulation.simplices]
    
    face_only_triangles = [triangle for triangle in triangles if not (set(triangle) - set(range(68)))]
    
    face1_partial = functools.partial(similarity_func, landmarks=face1_landmarks)
    face1_similarity = list(map(face1_partial, face_only_triangles))

    face2_partial = functools.partial(similarity_func, landmarks=face2_landmarks)
    face2_similarity = list(map(face2_partial, face_only_triangles))
    
    return abs(np.array(face1_similarity) - np.array(face2_similarity)), face_only_triangles