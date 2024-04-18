# import numpy as np

# # Clauser and Colleagues' Body Segment Parameters
# massFactor={"Hand":0.0065,
#             "Forearm":0.0161,
#             "Upper arm":0.0263,
#             "Forearm and hand":0.0227,
#             "Upper extremity":0.0490,
#             "Foot":0.0147,
#             "Leg":0.0435,
#             "Thigh":0.1027,
#             "Leg and foot":0.0582,
#             "Lower extremity":0.1610,
#             "Trunk":0.5070,
#             "Head":0.0728,
#             "Trunk and head":0.5801}
# lengthFactor={"Hand":0.1802,
#             "Forearm":0.3896,
#             "Upper arm":0.5130,
#             "Forearm and hand":0.6258,
#             "Upper extremity":0.4126,
#             "Foot":0.4485,
#             "Leg":0.3705,
#             "Thigh":0.1027,
#             "Leg and foot":0.4747,
#             "Lower extremity":0.3821,
#             "Trunk":0.3803,
#             "Head":0.4642,
#             "Trunk and head":0.5921}

# # Nodes number from Mediapipe are matched with body segments' name from Clauser and Colleagues (Research Methods in Biomechanics)
# nodeNumber={"Forearm":[[13,15],[14,16]],
#             "Upper arm":[[11,23],[12,24]],
#             "Leg":[[25,27],[26,28]],
#             "Thigh":[[23,25],[24,26]],
#             # "Trunk":[[11,23],[12,24]],
#             # "Head":[0]
#             }

# def calCalc(coord1,coord2,coord3,mass):
#     distance1=np.sqrt((coord2[0]-coord1[0])**2+(coord2[1]-coord1[1])**2) # scalar
#     distance2=np.sqrt((coord3[0]-coord2[0])**2+(coord3[1]-coord2[1])**2) # scalar
#     displacement=distance1+distance2 # scalar
#     velocity12=np.subtract(coord2,coord1)
#     velocity23=np.subtract(coord3,coord2)
#     acceleration=velocity23-velocity12 # have to be revised as there is gravitational field
#     acceleration=np.sqrt(acceleration[0]**2+acceleration[1]**2) # scalar
#     work=mass*acceleration*displacement # scalar
#     return work

# def segmentCG(proximal,distal,segName):
#     cg=np.add(distal,proximal)*lengthFactor[segName]+proximal
#     return cg

# def segmentMass(bodyWeight,segName):
#     segMass=massFactor[segName]*bodyWeight
#     return segMass

# def cgTrunk(coord):
#     proxTrunk=np.add(coord[11][1:3],coord[12][1:3])/2
#     distTrunk=np.add(coord[23][1:3],coord[24][1:3])/2
#     cg=segmentCG(proxTrunk,distTrunk,"Trunk")
#     return cg

# def calculate(coord1, coord2, coord3):
#     bodyWeight=60
#     cal=0

#     # Head 0
#     mHead=segmentMass(bodyWeight,"Head")
#     cgHead1=coord1[0][1:3]
#     cgHead2=coord2[0][1:3]
#     cgHead3=coord3[0][1:3]
#     cal+=calCalc(cgHead1,cgHead2,cgHead3,mHead)

#     # Trunk 11-23 12-24
#     mTrunk=segmentMass(bodyWeight,"Trunk")
#     cgTrunk1=cgTrunk(coord1)
#     cgTrunk2=cgTrunk(coord2)
#     cgTrunk3=cgTrunk(coord3)
#     cal+=calCalc(cgTrunk1,cgTrunk2,cgTrunk3,mTrunk)

#     for node in nodeNumber:
#         for pos in nodeNumber[node]:
#             prox=pos[0]
#             dist=pos[1]
#             mass=segmentMass(bodyWeight,node)
#             cg1=segmentCG(coord1[prox][1:3],coord1[dist][1:3],node)
#             cg2=segmentCG(coord2[prox][1:3],coord2[dist][1:3],node)
#             cg3=segmentCG(coord3[prox][1:3],coord3[dist][1:3],node)
#             cal+=calCalc(cg1,cg2,cg3,mass)

#     return cal

import numpy as np

massFactor = {"Hand": 0.0065,
              "Forearm": 0.0161,
              "Upper arm": 0.0263,
              "Forearm and hand": 0.0227,
              "Upper extremity": 0.0490,
              "Foot": 0.0147,
              "Leg": 0.0435,
              "Thigh": 0.1027,
              "Leg and foot": 0.0582,
              "Lower extremity": 0.1610,
              "Trunk": 0.5070,
              "Head": 0.0728,
              "Trunk and head": 0.5801}
lengthFactor = {"Hand": 0.1802,
                "Forearm": 0.3896,
                "Upper arm": 0.5130,
                "Forearm and hand": 0.6258,
                "Upper extremity": 0.4126,
                "Foot": 0.4485,
                "Leg": 0.3705,
                "Thigh": 0.1027,
                "Leg and foot": 0.4747,
                "Lower extremity": 0.3821,
                "Trunk": 0.3803,
                "Head": 0.4642,
                "Trunk and head": 0.5921}

nodeNumber = {"Forearm": [[13, 15], [14, 16]],
              "Upper arm": [[11, 23], [12, 24]],
              "Leg": [[25, 27], [26, 28]],
              "Thigh": [[23, 25], [24, 26]]}

bodyWeight = 60

def calCalc(coord1, coord2, coord3, node, fps):
    mass=massFactor[node] * bodyWeight
    coord1 = np.array(coord1)
    coord2 = np.array(coord2)
    coord3 = np.array(coord3)
    distance1 = np.linalg.norm(coord2 - coord1)
    distance2 = np.linalg.norm(coord3 - coord2)
    displacement = (distance1 + distance2)/1000
    velocity12 = (coord2 - coord1)*fps
    velocity23 = (coord3 - coord2)*fps
    acceleration = np.linalg.norm(velocity23 - velocity12)*fps/1000
    work = mass * acceleration * displacement
    return work

def segmentCG(proximal, distal, segName):
    proximal = np.array(proximal)
    distal = np.array(distal)
    return (distal + proximal) * lengthFactor[segName] + proximal

# def segmentMass(bodyWeight, segName):
#     return massFactor[segName] * bodyWeight

def cgTrunk(coord):
    proxTrunk = np.mean([coord[11][1:3], coord[12][1:3]], axis=0)
    distTrunk = np.mean([coord[23][1:3], coord[24][1:3]], axis=0)
    return segmentCG(proxTrunk, distTrunk, "Trunk")

def calculate(coord1, coord2, coord3, fps):
    cal = 0

    # Head
    # mHead = segmentMass(bodyWeight, "Head")
    cgHead1, cgHead2, cgHead3 = coord1[0][1:3], coord2[0][1:3], coord3[0][1:3]
    cal += calCalc(cgHead1, cgHead2, cgHead3, "Head", fps)

    # Trunk
    # mTrunk = segmentMass(bodyWeight, "Trunk")
    cgTrunk1, cgTrunk2, cgTrunk3 = cgTrunk(coord1), cgTrunk(coord2), cgTrunk(coord3)
    cal += calCalc(cgTrunk1, cgTrunk2, cgTrunk3, "Trunk", fps)

    for node, pos in nodeNumber.items():
        for prox, dist in pos:
            # mass = segmentMass(bodyWeight, node)
            cg1 = segmentCG(coord1[prox][1:3], coord1[dist][1:3], node)
            cg2 = segmentCG(coord2[prox][1:3], coord2[dist][1:3], node)
            cg3 = segmentCG(coord3[prox][1:3], coord3[dist][1:3], node)
            cal += calCalc(cg1, cg2, cg3, node, fps)

    return cal