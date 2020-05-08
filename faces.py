import cv2
import numpy as np
from morphing import EDGE_LANDMARKS

sam = cv2.cvtColor(cv2.imread('./my-images/sam.jpg'), cv2.COLOR_BGR2RGB)
sam_cropped = sam[950:2950, 650:2150, :].copy()
sam_resized = cv2.resize(sam_cropped, (900, 1200))

ryan = cv2.cvtColor(cv2.imread('./my-images/ryan.jpg'), cv2.COLOR_BGR2RGB)
ryan_cropped = ryan[150:2650, 563:2438, :].copy()
ryan_resized = cv2.resize(ryan_cropped, (900, 1200))

rachelle = cv2.cvtColor(cv2.imread('./my-images/rachelle.jpg'), cv2.COLOR_BGR2RGB)
rachelle_cropped = rachelle[500:3200, 460:2485, :].copy()
rachelle_resized = cv2.resize(rachelle_cropped, (900, 1200))

jared = cv2.cvtColor(cv2.imread('./my-images/jared.jpg'), cv2.COLOR_BGR2RGB)
jared_cropped = jared[550:2950, 550:2350, :].copy()
jared_resized = cv2.resize(jared_cropped, (900, 1200))

amanda = cv2.cvtColor(cv2.imread('./my-images/amanda.jpg'), cv2.COLOR_BGR2RGB)
amanda_cropped = amanda[650:2250, 925:2125, :].copy()
amanda_resized = cv2.resize(amanda_cropped, (900, 1200))

darren = cv2.cvtColor(cv2.imread('./my-images/darren.jpg'), cv2.COLOR_BGR2RGB)
darren_cropped = darren[800:2300, 825:1950, :].copy()
darren_resized = cv2.resize(darren_cropped, (900, 1200))

linda = cv2.cvtColor(cv2.imread('./my-images/linda.jpg'), cv2.COLOR_BGR2RGB)
linda_cropped = linda[500:3000, 600:2475, :].copy()
linda_resized = cv2.resize(linda_cropped, (900, 1200))

roger = cv2.cvtColor(cv2.imread('./my-images/roger.jpg'), cv2.COLOR_BGR2RGB)
roger_cropped = roger[500:3300, 450:2550, :].copy()
roger_resized = cv2.resize(roger_cropped, (900, 1200))

brittany = cv2.cvtColor(cv2.imread('./my-images/britt.jpg'), cv2.COLOR_BGR2RGB)
brittany_cropped = brittany[350:2450, 380:1955, :].copy()
brittany_resized = cv2.resize(brittany_cropped, (900, 1200))

steph = cv2.cvtColor(cv2.imread('./my-images/steph.jpg'), cv2.COLOR_BGR2RGB)
steph_cropped = steph[250:1050, 225:825, :].copy()
steph_resized = cv2.resize(steph_cropped, (900, 1200))

nico = cv2.cvtColor(cv2.imread('./my-images/nico.jpg'), cv2.COLOR_BGR2RGB)
nico_cropped = nico[800:2950, 590:2200, :].copy()
nico_resized = cv2.resize(nico_cropped, (900, 1200))

ryano = cv2.cvtColor(cv2.imread('./my-images/ryano.jpg'), cv2.COLOR_BGR2RGB)
ryano_cropped = ryano[1200:2600, 900:1950, :].copy()
ryano_resized = cv2.resize(ryano_cropped, (900, 1200))

peter = cv2.cvtColor(cv2.imread('./my-images/peter.jpg'), cv2.COLOR_BGR2RGB)
peter_cropped = peter[900:3100, 550:2200, :].copy()
peter_resized = cv2.resize(peter_cropped, (900, 1200))

conchi = cv2.cvtColor(cv2.imread('./my-images/conchi.jpg'), cv2.COLOR_BGR2RGB)
conchi_cropped = conchi[400:3100, 425:2450, :].copy()
conchi_resized = cv2.resize(conchi_cropped, (900, 1200))

mii_ryan = cv2.cvtColor(cv2.imread('./my-images/mii-ryan.png'), cv2.COLOR_BGR2RGB)
mii_ryan_cropped = mii_ryan[140:635, 235:606].copy()
mii_ryan_resized = cv2.resize(mii_ryan_cropped, (900, 1200))

mii_sam = cv2.cvtColor(cv2.imread('./my-images/mii-sam.png'), cv2.COLOR_BGR2RGB)
mii_sam_cropped = mii_sam[140:635, 135:506].copy()
mii_sam_resized = cv2.resize(mii_sam_cropped, (900, 1200))

with open('./my-images/mii-ryan-landmarks.txt', 'r') as filehandle:
    test_landmarks = filehandle.read().split('\n')
mii_ryan_landmarks = [[float(round(float(x))) for x in landmark[1:-1].split(', ')] for landmark in test_landmarks if landmark]

landmarks_plus_edge = np.zeros((len(mii_ryan_landmarks) + len(EDGE_LANDMARKS), 2))
landmarks_plus_edge[0:len(mii_ryan_landmarks), :] = mii_ryan_landmarks.copy()
landmarks_plus_edge[len(mii_ryan_landmarks):, :] = EDGE_LANDMARKS
mii_ryan_landmarks = landmarks_plus_edge

with open('./my-images/mii-sam-landmarks.txt', 'r') as filehandle:
    test_landmarks = filehandle.read().split('\n')
mii_sam_landmarks = [[float(round(float(x))) for x in landmark[1:-1].split(', ')] for landmark in test_landmarks if landmark]

landmarks_plus_edge = np.zeros((len(mii_sam_landmarks) + len(EDGE_LANDMARKS), 2))
landmarks_plus_edge[0:len(mii_sam_landmarks), :] = mii_sam_landmarks.copy()
landmarks_plus_edge[len(mii_sam_landmarks):, :] = EDGE_LANDMARKS
mii_sam_landmarks = landmarks_plus_edge