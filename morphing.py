import cv2
import dlib
import numpy as np
import scipy as sp
from scipy.spatial import Delaunay
from matplotlib import pyplot as plt
from scipy.sparse import linalg, csr_matrix, lil_matrix
import os
from utils import *


FACE_DETECTOR = dlib.get_frontal_face_detector()
FACE_PREDICTOR = predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

EDGE_LANDMARKS = [[0, 0],
                  [175, 0],
                  [350, 0],
                  [550, 0],
                  [725, 0],
                  [899, 0],
                  [0, 200],
                  [0, 400],
                  [0, 600],
                  [0, 800],
                  [0, 1000],
                  [0, 1199],
                  [175, 1199],
                  [350, 1199],
                  [550, 1199],
                  [725, 1199],
                  [899, 1199],
                  [899, 200],
                  [899, 400],
                  [899, 600],
                  [899, 800],
                  [899, 1000]]


def get_face_landmarks(face_image, detector=FACE_DETECTOR, predictor=FACE_PREDICTOR, edges=EDGE_LANDMARKS):
    face = detector(face_image, 1)
    shape = predictor(face_image, face[0])
    
    NUM_LANDMARKS = 68
    landmarks = np.zeros((NUM_LANDMARKS, 2))
    for idx in range(NUM_LANDMARKS):
        landmarks[idx] = [shape.part(idx).x, shape.part(idx).y]
    
    if edges:
        landmarks_plus_edge = np.zeros((NUM_LANDMARKS + len(edges), 2))
        landmarks_plus_edge[0:68, :] = landmarks.copy()
        landmarks_plus_edge[68:, :] = edges
        landmarks = landmarks_plus_edge
        
    return landmarks


def get_image_mask(W, H, vertices):
    mask = np.zeros((H, W, 3)).astype('uint8')
    mask = cv2.fillPoly(mask, [vertices], (1, 1, 1))
    return mask


def get_transformations_and_masks(triangulation, average_landmarks, original_landmarks):
    transformations = {}
    masks = {}
    for idx, triangle in enumerate(triangulation.simplices):
        orig_verts = np.take(original_landmarks, triangle, axis=0).astype('int32')
        average_verts = np.take(average_landmarks, triangle, axis=0).astype('int32')

        A = lil_matrix((6,6))
        b = np.zeros((6, 1))
        equation = 0
        for idx2, vert in enumerate(orig_verts):
            A[equation, :] = [vert[0], vert[1], 1, 0, 0, 0]
            A[equation+1, :] = [0, 0, 0, vert[0], vert[1], 1]
            b[equation] = average_verts[idx2][0]
            b[equation+1] = average_verts[idx2][1]
            equation += 2

        A = A.tocsr()
        results = linalg.lsqr(A, b, atol=1e-13, btol=1e-13)
        v = results[0]
        M = v.reshape((2,3))
        transformations[idx] = M

        mask = get_image_mask(900, 1200, orig_verts)
        masks[idx] = mask
    return transformations, masks


def get_average_face(face1, face2, weight=0.5, face1_landmarks=None, face2_landmarks=None):
    if face1_landmarks is None: 
        face1_landmarks = get_face_landmarks(face1)
    if face2_landmarks is None:
        face2_landmarks = get_face_landmarks(face2)
    
    average_landmarks = np.average([face1_landmarks, face2_landmarks], axis=0, weights=[weight, 1.0 - weight])
    
    triangulation = Delaunay(average_landmarks)
    
    face1_transformations, face1_masks = get_transformations_and_masks(triangulation,
                                                                       average_landmarks,
                                                                       face1_landmarks)
    face2_transformations, face2_masks = get_transformations_and_masks(triangulation,
                                                                       average_landmarks,
                                                                       face2_landmarks)
    
    face1_morphed = np.zeros(face1.shape)
    for idx in range(len(face1_masks)):
        piece = face1 * face1_masks[idx][:, :, :]
        morphed_piece = cv2.warpAffine(piece, face1_transformations[idx], (900, 1200))
        face1_morphed = blendImages(morphed_piece, face1_morphed)
    face1_morphed = cv2.medianBlur(face1_morphed, 5)

    face2_morphed = np.zeros(face2.shape)
    for idx in range(len(face2_masks)):
        piece = face2 * face2_masks[idx][:, :, :]
        morphed_piece = cv2.warpAffine(piece, face2_transformations[idx], (900, 1200))
        face2_morphed = blendImages(morphed_piece, face2_morphed)
    face2_morphed = cv2.medianBlur(face2_morphed, 5)
    
    average_face = np.average([face1_morphed, face2_morphed],
                              axis=0,
                              weights=[weight, 1.0 - weight])
    average_face = average_face.astype('uint8')
    
    return average_face, face1_morphed, face2_morphed


def get_face_morphing_sequence(face1, face2, frames=120, face1_landmarks=None, face2_landmarks=None):
    images = []
    for weight in np.linspace(0, 1, frames):
        print(f'Computing average face with weight {weight}')
        weighted_average_face, _, _ = get_average_face(face1,
                                                       face2,
                                                       weight=weight,
                                                       face1_landmarks=face1_landmarks,
                                                       face2_landmarks=face2_landmarks)
        images.append(weighted_average_face)
    return images