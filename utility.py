import numpy as np

'''
    References:
    ----------
        [1] R.M. Haralick, K. Shanmugam, and I. Dinstein, "Textural features for
           image classification", IEEE Transactions on Systems, Man, and
           Cybernetics, vol. SMC-3, no. 6, pp. 610-621, Nov. 1973.
           https://doi.org/10.1109/TSMC.1973.4309314
'''


def glcm_neighbour(x1, y1, d, angle):
    '''
    get neoighbour location based on distance and angle
    '''
    x2 = x1
    y2 = y1
    if angle == 0:        
        y2 = y1 + d
    elif angle == 45:
        x2 = x1 - d
        y2 = y1 + d
    elif angle == 90:
        x2 = x1 - d
    elif angle == 135:
        x2 = x1 - d
        y2 = y1 - d

    return x2, y2
    
def construct_glcm(img, gray_levels, d, theta, symmetric=False, normalized=False):
    '''
    construct glcm base on
        gray_levels: for grayscale image it will be 256
        d: distance, example: 1, 2, 5
        theta: angle, example: 0, 45, 90
        symmetric: meaning if the result will be symmetic
        normalized: meaning if the result needs to be normalized

        returns glcm
    '''
    glcm = np.zeros(shape=(gray_levels,gray_levels), dtype="uint8")
    rows, cols = img.shape
    img_pad = np.pad(img, pad_width=d, mode='constant', constant_values=-1).astype(np.int8)
    for x in range(rows):
        for y in range(cols):
            x1 = x + d
            y1 = y + d
            x2, y2 = glcm_neighbour(x1, y1, d, theta)
            if img_pad[x2, y2] != -1:
                glcm[img_pad[x1, y1], img_pad[x2, y2]] += 1 
    if symmetric:
        glcm = glcm + glcm.T
    if normalized:
        total_freq = np.sum(glcm)
        glcm = glcm / total_freq
    return glcm
    
def sum(glcm):
    '''
    sum over x or y axis
    '''
    Sx = np.sum(glcm, axis=0)
    Sy = np.sum(glcm, axis=1)
    return Sx, Sy
    
def mean(glcm):
    '''
    mean over x or y axis
    '''
    l = glcm.shape[0]
    Sx, Sy = sum(glcm)

    add_res = 0
    for i in range(l):
        add_res += Sx[i] * (i)        
    Ux = add_res 

    add_res = 0
    for i in range(l):
        add_res += Sy[i] * (i)        
    Uy = add_res 

    return Ux, Uy

def sigma(glcm):
    '''
    standard deviation over x or y axis
    '''
    l = glcm.shape[0]
    
    Sx, Sy = sum(glcm)
    Ux, Uy = mean(glcm)

    add_res = 0
    for i in range(l):
        add_res += Sx[i] * ((i - Ux) ** 2)
    SiGx = np.sqrt(add_res)

    add_res = 0
    for i in range(l):
        add_res += Sy[i] * ((i - Uy) ** 2)
    SiGy = np.sqrt(add_res)

    return SiGx, SiGy


def energy(glcm):
    return np.sum(glcm**2)

def contrast(glcm):
    _contrast = 0.0
    lx, ly = glcm.shape
    for i in range(lx):
        for j in range(ly):
            _contrast += (i - j) ** 2 * glcm[i,j]
    return _contrast 

def correlation(glcm):
    _correlation = 0.0
    lx, ly = glcm.shape
    Ux, Uy = mean(glcm)
    SiGx, SiGy = sigma(glcm)
    for i in range(lx):
        for j in range(ly):
            _correlation += (i - Ux) * (j - Uy) * glcm[i,j]
    return _correlation / (SiGx * SiGy)

def entropy(glcm):
    _entropy = 0.0
    lx, ly = glcm.shape
    epsilon = np.finfo(float).eps
    for i in range(lx):
        for j in range(ly):
            _entropy +=  glcm[i,j] * np.log(epsilon if glcm[i,j] == 0 else glcm[i,j])
    return -_entropy

def IDM(glcm):
    _idm = 0.0
    lx, ly = glcm.shape
    for i in range(lx):
        for j in range(ly):
            _idm +=  glcm[i,j] / (1 + (i - j) ** 2)
    return _idm

