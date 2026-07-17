import numpy as np

def skew_symmetric(v):
    v = np.array(v).flatten()
    result = np.array([[0, -v[2], v[1]],
                    [v[2], 0, -v[0]],
                    [-v[1], v[0], 0]])

    return result

# Directional Cosine Matrix to (3-2-1) Euler Angles
def dcm_to_ea321(dcm):

    # allocate parameters
    theta1 = np.arctan2(dcm[0,1], dcm[0,0])
    theta2 = -np.arcsin(dcm[0,2])
    theta3 = np.arctan2(dcm[1,2], dcm[2,2])

    euler_angles_rad = np.array([theta1, theta2, theta3]).reshape(3,1)

    return euler_angles_rad

# Directional Cosine Matrix to (3-1-3) Euler Angles
def dcm_to_ea313(dcm):

    # allocate parameters
    theta1 = np.arctan2(dcm[2,0], -dcm[2,1])
    theta2 = np.arccos(dcm[2,2])
    theta3 = np.arctan2(dcm[0,2], dcm[1,2])

    euler_angles_rad = np.array([theta1, theta2, theta3]).reshape(3,1)

    return euler_angles_rad
    
# Directional Cosine Matrix to Quaternions (Sheppard's method)
def dcm_to_quaternion(dcm):

    # allocate parameters
    b0_square = (1 + np.trace(dcm)) / 4
    b1_square = (1 + 2 * dcm[0,0] - np.trace(dcm)) / 4
    b2_square = (1 + 2 * dcm[1,1] - np.trace(dcm)) / 4
    b3_square = (1 + 2 * dcm[2,2] - np.trace(dcm)) / 4

    # find max_index of the largest b_square
    b_square = np.array([b0_square, b1_square, b2_square, b3_square])
    max_index = np.argmax(b_square)

    if max_index == 0:
        b0 = np.sqrt(max(0, b0_square))
        b1 = (dcm[1,2] - dcm[2,1]) / (4 * b0)
        b2 = (dcm[2,0] - dcm[0,2]) / (4 * b0)
        b3 = (dcm[0,1] - dcm[1,0]) / (4 * b0)

    elif max_index == 1:
        b1 = np.sqrt(max(0, b1_square))
        b0 = (dcm[1,2] - dcm[2,1]) / (4 * b1)
        b2 = (dcm[0,1] + dcm[1,0]) / (4 * b1)
        b3 = (dcm[2,0] + dcm[0,2]) / (4 * b1)

    elif max_index == 2:
        b2 = np.sqrt(max(0, b2_square))
        b0 = (dcm[2,0] - dcm[0,2]) / (4 * b2)
        b1 = (dcm[0,1] + dcm[1,0]) / (4 * b2)
        b3 = (dcm[1,2] + dcm[2,1]) / (4 * b2)

    elif max_index == 3:
        b3 = np.sqrt(max(0, b3_square))
        b0 = (dcm[0,1] - dcm[1,0]) / (4 * b3)
        b1 = (dcm[2,0] + dcm[0,2]) / (4 * b3)
        b2 = (dcm[1,2] + dcm[2,1]) / (4 * b3)

    if b0 < 0:
        b0, b1, b2, b3 = -b0, -b1, -b2, -b3
        
    quaternions = np.array([b0, b1, b2, b3]).reshape(4,1)

    return quaternions

# Directional Cosine Matrix to CLassical Rodrigues Parameters
def dcm_to_crp(dcm):
    zeta_square = 1 + np.trace(dcm)
    q = np.array([dcm[1,2] - dcm[2,1], dcm[2,0] - dcm[0,2], dcm[0,1] - dcm[1,0]]).reshape(3,1) / zeta_square

    return q

# Directional Cosine Matrix to Modified Rodrigues Parameters
def dcm_to_mrp(dcm):
    zeta = np.sqrt(1 + np.trace(dcm))
    sigma = np.array([dcm[1,2] - dcm[2,1], dcm[2,0] - dcm[0,2], dcm[0,1] - dcm[1,0]]).reshape(3,1) / (zeta*(zeta + 2))

    return sigma

# (3-2-1) Euler Angles to Directional Cosine Matrix
def ea321_to_dcm(euler_angles_rad):

    # allocate parameters
    theta1, theta2, theta3 = euler_angles_rad.flatten()

    # refactoring
    c1, s1 = np.cos(theta1), np.sin(theta1)
    c2, s2 = np.cos(theta2), np.sin(theta2)
    c3, s3 = np.cos(theta3), np.sin(theta3)

    dcm = np.array([[c2*c1, c2*s1, -s2],
                    [s3*s2*c1 - c3*s1, s3*s2*s1 + c3*c1, s3*c2],
                    [c3*s2*c1 + s3*s1, c3*s2*s1 - s3*c1, c3*c2]])

    return dcm

# (3-1-3) Euler Angles to Directional Cosine Matrix
def ea313_to_dcm(euler_angles_rad):

    # allocate parameters
    theta1, theta2, theta3 = euler_angles_rad.flatten()

    # refactoring
    c1, s1 = np.cos(theta1), np.sin(theta1)
    c2, s2 = np.cos(theta2), np.sin(theta2)
    c3, s3 = np.cos(theta3), np.sin(theta3)

    dcm = np.array([[c3*c1 - s3*c2*s1, c3*s1 + s3*c2*c1, s3*s2],
                    [-s3*c1 - c3*c2*s1, -s3*s1 + c3*c2*c1, c3*s2],
                    [s2*s1, -s2*c1, c2]])

    return dcm

# Quaternions to Directional Cosine Matrix
def quaternion_to_dcm(quaternions):

    # allocate parameters (b0 is a scalar part)
    b0, b1, b2, b3 = quaternions.flatten()

    dcm = np.array([[b0**2 + b1**2 - b2**2 - b3**2,  2*(b1*b2 + b0*b3), 2*(b1*b3 - b0*b2)],
                    [2*(b1*b2 - b0*b3), b0**2 - b1**2 + b2**2 - b3**2,  2*(b2*b3 + b0*b1)],
                    [2*(b1*b3 + b0*b2), 2*(b2*b3 - b0*b1), b0**2 - b1**2 - b2**2 + b3**2]])
        
    return dcm

# Classical Rodrigues Parameters to Directional Cosine Matrix
def crp_to_dcm(q):
    dcm = ((1 - np.vdot(q, q)) * np.eye(3) + 2 * np.outer(q, q) - 2 * skew_symmetric(q)) / (1 + np.vdot(q, q))

    return dcm

# Modified Rodrigues Parameters to Directional Cosine Matrix
def mrp_to_dcm(sigma):
    dcm = (np.eye(3) + (8 * skew_symmetric(sigma) @ skew_symmetric(sigma) - 4 * (1 - np.vdot(sigma,sigma)) * skew_symmetric(sigma)) / (1 + np.vdot(sigma,sigma))**2)

    return dcm


class Attitude:
    
    # the most fundamental attitude coordinate is dcm
    def __init__(self, dcm):
        self.dcm = dcm

    @classmethod
    def from_ea321(cls, euler_angles_rad):
        dcm = ea321_to_dcm(euler_angles_rad)

        return cls(dcm)

    @classmethod
    def from_ea313(cls, euler_angles_rad):
        dcm = ea313_to_dcm(euler_angles_rad)
        
        return cls(dcm)
    
    @classmethod
    def from_quaternion(cls, quaternions):
        dcm = quaternion_to_dcm(quaternions)
        
        return cls(dcm)

    @classmethod
    def from_crp(cls, q):
        dcm = crp_to_dcm(q)

        return cls(dcm)
    
    @classmethod
    def from_mrp(cls, sigma):
        dcm = mrp_to_dcm(sigma)

        return cls(dcm) 

    def get_ea321(self):
        ea321 = self.dcm_to_ea321(self.dcm)

        return ea321
    
    def get_ea313(self):
        ea313 = self.dcm_to_ea313(self.dcm)

        return ea313
    
    def get_quaternion(self):
        quaternion = self.dcm_to_quaternion(self.dcm)

        return quaternion

    def get_crp(self):
        crp = self.dcm_to_crp(self.dcm)

        return crp
    
    def get_mrp(self):
        mrp = self.dcm_to_mrp(self.dcm)

        return mrp