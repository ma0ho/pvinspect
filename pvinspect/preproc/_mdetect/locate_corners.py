import numpy as np
from scipy.interpolate import interpn
import matplotlib.pyplot as plt
from skimage.transform.integral import integral_image
from scipy.ndimage.filters import gaussian_filter1d
from .config import GAUSSIAN_RELATIVE_SIGMA
from .summary import Summary

summary = Summary('locate_corners')

def inner_corners_model(n_cols, m_rows):
    x, y = np.meshgrid(np.arange(1, n_cols), np.arange(1, m_rows))
    return np.array([x.flatten(), y.flatten()], dtype = np.float64).T

def outer_corners_model(n_cols, m_rows):
    x = list(range(0, n_cols+1)) + [n_cols]*(m_rows-1) + list(reversed(range(0, n_cols+1))) + [0]*(m_rows-1)
    y = [0]*(n_cols+1) + list(range(1, m_rows)) + [m_rows]*(n_cols+1) + list(reversed(range(1, m_rows)))
    
    return np.array([x, y], dtype = np.float64).T

def _corner_patches(coords, patch_size):
    d = patch_size/2
    coords = np.repeat(np.expand_dims(coords, 1), 4, 1)
    coords[:,0] -= d
    coords[:,1,0] += d
    coords[:,1,1] -= d
    coords[:,2] += d
    coords[:,3,0] -= d
    coords[:,3,1] += d
    return coords

#def _extract_profile(img, p0, p1):
#    n_samples = int(np.ceil(np.linalg.norm(p1-p0)))
#    x = np.linspace(p0[0], p1[0], n_samples)
#    y = np.linspace(p0[1], p1[1], n_samples)
#    yx = np.array([y, x]).T
#    return interpn((np.arange(img.shape[0]), np.arange(img.shape[1])), img, yx, method = 'linear')

#def _bivariate_polynomial(x, y, d):
#    monomials = []
#    for i in range(d+1):
#        for j in range(d+1):
#            monomials.append(x**i * y**j)
#    return np.array(monomials).T

def _extract_profile2d(img, patch, transform, n_samples):
    n_samples = 80
    x, y = np.meshgrid(
        np.linspace(patch[0,0], patch[1,0], n_samples),
        np.linspace(patch[0,1], patch[3,1], n_samples))
    xy = np.array([x.flatten(), y.flatten()]).T
    xy_t = transform(xy)
    yx_t = np.flip(xy_t, 1)
    v = interpn((np.arange(img.shape[0]), np.arange(img.shape[1])), img, yx_t, method = 'linear', bounds_error = False, fill_value = 0.0)
    summary.put('patch_{:d}_{:d}'.format(int(patch[0,0]+1), int(patch[0,1]+1)), v.reshape(n_samples, n_samples))
    return x, y, v.reshape(n_samples, n_samples)

def _find_stops(img, dim, sigma, strategy, patch, edgetype = 'ridge'):
    img = img.T if dim == 0 else img

    #plt.imshow(img)
    #plt.show()

    # calculate downsampling
    size = 0.5*np.sum(img.shape)

    # extract profile of cumsum along di
    profile = np.sum(img, 1)
    profile_smooth = gaussian_filter1d(profile, sigma)
    profile_smooth = profile_smooth-np.min(profile_smooth)
    profile_smooth = profile_smooth/(np.max(profile_smooth)+1e-5)

    # calculate gradient of that
    grad_smooth = np.gradient(profile_smooth)

    summary.put('patch_{:d}_{:d}_profile_{}'.format(int(patch[0,0]+1), int(patch[0,1]+1), 'x' if dim == 0 else 'y'), profile)
    summary.put('patch_{:d}_{:d}_profile_grad_{}'.format(int(patch[0,0]+1), int(patch[0,1]+1), 'x' if dim == 0 else 'y'), grad_smooth)

    # find maxima
    # everything that is > 1.5*std(grad_smooth) is considered a maximum
    grad_smooth_t = (grad_smooth > 1.5*np.std(grad_smooth))*grad_smooth
    maxima = []
    while True:
        m = np.argmax(grad_smooth_t)

        # go to the right
        for i in range(grad_smooth_t.shape[0]-m):
            if grad_smooth_t[m+i] > 0:
                grad_smooth_t[m+i] = 0
            else:
                break

        # go to the left
        for i in range(1,m):
            if grad_smooth_t[m-i] > 0:
                grad_smooth_t[m-i] = 0
            else:
                break
        maxima.append(m)

        if np.sum(grad_smooth_t) == 0:
            break
    
    # find minima
    # everything that is < -1.5*std(grad_smooth) is considered a minimum
    grad_smooth_t = (grad_smooth < -1.5*np.std(grad_smooth))*grad_smooth
    minima = []
    while True:
        m = np.argmin(grad_smooth_t)
        for i in range(grad_smooth_t.shape[0]-m):
            if grad_smooth_t[m+i] < 0:
                grad_smooth_t[m+i] = 0
            else:
                break
        for i in range(1,m):
            if grad_smooth_t[m-i] < 0:
                grad_smooth_t[m-i] = 0
            else:
                break
        minima.append(m)

        if np.sum(grad_smooth_t) == 0:
            break

    # skip if no minima/maxima found
    if len(maxima) == 0 or len(minima) == 0:
        return None

    # for "ridge" edges, we require that the number of minima and maxima is consistent
    if len(maxima) != len(minima) and edgetype == 'ridge':
        return None

    # go for maxima (edgetype=ridge/up) or minima (edgetype=down)
    target = np.array(maxima) if edgetype in ('ridge', 'up') else np.array(minima)

    # determine correct extremum by spatial order
    if strategy == 'order':
        target = np.sort(target)
        if target.shape[0]%2 == 1:
            # take middle
            pos = target[target.shape[0]//2]
        else:
            # take two mid target and select by maximum abs value
            m1 = target[target.shape[0]//2 -1]
            m2 = target[target.shape[0]//2]
            if np.abs(grad_smooth[m1]) > np.abs(grad_smooth[m2]):
                pos = m1
            else:
                pos = m2
        #print(pos)
        #plt.plot(np.arange(profile.shape[0]), grad_smooth)
        #plt.show()

    # determine correct extremum by distance to center of the patch
    elif strategy == 'distance':
        center = profile_smooth.shape[0]/2
        dist = np.abs(target-center)
        idx = np.argmin(dist)
        pos = target[idx]
        #plt.plot(np.arange(profile.shape[0]), grad_smooth)
        #plt.show()

    # take closest minimum (edgetype=ridge) or skip
    if edgetype == 'ridge':
        minima = np.array(minima)
        dist = pos-minima

        # minimum needs to be right of maximum
        if np.sum(dist > 0) == 0:
            return None

        min_idx = np.argmin(dist[dist>0])
        minimum = minima[dist>0][min_idx]

    summary.put('patch_{:d}_{:d}_pos_{}'.format(int(patch[0,0]+1), int(patch[0,1]+1), 'x' if dim == 0 else 'y'), pos)
    
    # for corner edges, we only get one estimate
    if edgetype in ('up', 'down'):
        return pos

    # for ridges, we get the left and right boundary of the ridge
    elif edgetype == 'ridge':
        return [minimum, pos]

def _determine_sampling_and_filter(patch_t):
    patch_width = max(np.linalg.norm(patch_t[1]-patch_t[0]), np.linalg.norm(patch_t[3]-patch_t[2]))
    patch_height = max(np.linalg.norm(patch_t[2]-patch_t[1]), np.linalg.norm(patch_t[3]-patch_t[0]))
    n_samples = max(patch_height, patch_width)

    sigma = GAUSSIAN_RELATIVE_SIGMA*n_samples
    s = np.floor(2/3 * np.pi * sigma)

    if s > 1:
        sigma_s = (3*s) / (2*np.pi)
        if sigma**2 - sigma_s**2 < 0.5**2:
            # convince Nyquist
            s -= 1
            sigma_s = (3*s) / (2*np.pi)
        sigma_filter = np.sqrt(sigma**2 - sigma_s**2)
        n_samples /= s
    else:
        sigma_filter = sigma

    return int(n_samples), sigma_filter

def fit_inner_corners(img, coords, transform, patch_size):
    patches = _corner_patches(coords, patch_size)
    accepted_flags = list()
    corners = list()

    # determine #samples and filter for one example only
    n_samples, sigma_filter = _determine_sampling_and_filter(transform(patches[len(patches)//2]))

    for i in range(patches.shape[0]):
        _, _, v = _extract_profile2d(img, patches[i], transform, n_samples)
        
        # find stop in x and y direction
        xstops = _find_stops(v, 0, sigma_filter, 'order', patches[i])
        ystops = _find_stops(v, 1, sigma_filter, 'order', patches[i])

        # if both succeded
        if xstops is not None and ystops is not None:

            # xstops and ystops is given in patch coordinates -> transform to image coordinates
            x_size, y_size = v.shape[1], v.shape[0]
            xstops[0] /= x_size
            xstops[1] /= x_size
            ystops[0] /= y_size
            ystops[1] /= y_size

            # calculate corner coordinate
            xstopdiff = xstops[1] - xstops[0]
            ystopdiff = ystops[1] - ystops[0]
            corner = [xstops[0] + xstopdiff/2, ystops[0] + ystopdiff/2]

            ## check if 
            #reject_thres = 1/patch_size * 0.1

            #accepted = xstopdiff > 0 and ystopdiff > 0 and xstopdiff < reject_thres and ystopdiff < reject_thres
            
            #accepted_flags.append(accepted)
            accepted_flags.append(True)
            corners.append(corner)
        else:
            accepted_flags.append(False)
            corners.append([0,0])

        #plt.imshow(v, cmap = plt.get_cmap('gray'))

        #if accepted:
        #    plt.plot([corner[0]*x_size], [corner[1]*y_size], marker = 'x', color = 'green')
        #else:
        #    plt.plot([corner[0]*x_size], [corner[1]*y_size], marker = 'x', color = 'red')

        #plt.show()
        #plt.clf()

    corners = np.array(corners)
    corners = corners-(patch_size/2)
    return coords+corners, np.array(accepted_flags)


def fit_outer_corners(img, coords, transform, patch_size, n_cols, m_rows):
    patches = _corner_patches(coords, patch_size)
    accepted_flags = list()
    corners = list()

    # determine #samples and filter for one example only
    n_samples, sigma_filter = _determine_sampling_and_filter(transform(patches[len(patches)//2]))

    for i in range(patches.shape[0]):
        _, _, v = _extract_profile2d(img, patches[i], transform, n_samples)
        #plt.imshow(v, cmap='gray')
        #plt.show()
        
        if i <= n_cols:
            y_edgetype = 'up'
        elif i >= n_cols + m_rows and i <= 2*n_cols + m_rows:
            y_edgetype = 'down'
        else:
            y_edgetype = 'ridge'
        
        if i >= n_cols and i <= n_cols+m_rows:
            x_edgetype = 'down'
        elif i >= 2*n_cols + m_rows or i == 0:
            x_edgetype = 'up'
        else:
            x_edgetype = 'ridge'

        xstops = _find_stops(v, 0, sigma_filter, 'distance', patches[i], x_edgetype)
        ystops = _find_stops(v, 1, sigma_filter, 'distance', patches[i], y_edgetype)

        if xstops is not None and ystops is not None:
            x_size, y_size = v.shape[1], v.shape[0]

            xc = ((xstops[1]-xstops[0])/2)+xstops[0] if x_edgetype == 'ridge' else xstops
            yc = ((ystops[1]-ystops[0])/2)+ystops[0] if y_edgetype == 'ridge' else ystops
            xc /= x_size
            yc /= y_size

            #xstop /= x_size
            #ystop /= y_size

            #xstops[0] /= x_size
            #xstops[1] /= x_size
            #ystops[0] /= y_size
            #ystops[1] /= y_size
            
            #xstopdiff = xstops[1] - xstops[0] if x_edgetype == 'ridge' else 0
            #ystopdiff = ystops[1] - ystops[0] if y_edgetype == 'ridge' else 0

            #xc = xstops[0] + xstopdiff/2 if x_edgetype != 'up' else xstops[1]
            #yc = ystops[0] + ystopdiff/2 if y_edgetype != 'up' else ystops[1]

            corner = [xc, yc]

            reject_thres = 1/patch_size * 0.1

            accepted = True
            #if x_edgetype == 'ridge' and (xstops[1]-xstops[0])/x_size > reject_thres:
            #    accepted = False
            #if y_edgetype == 'ridge' and (ystops[1]-ystops[0])/y_size > reject_thres:
            #    accepted = False
            
            #accepted = xstopdiff >= 0 and ystopdiff >= 0 and xstopdiff < reject_thres and ystopdiff < reject_thres
            
            accepted_flags.append(accepted)
            corners.append(corner)
        else:
            accepted_flags.append(False)
            corners.append([0,0])

        #plt.imshow(v, cmap = plt.get_cmap('gray'))

        #if accepted:
        #    plt.plot([corner[0]*x_size], [corner[1]*y_size], marker = 'x', color = 'green')
        #else:
        #    plt.plot([corner[0]*x_size], [corner[1]*y_size], marker = 'x', color = 'red')

        #plt.show()
        #plt.clf()
    
    corners = np.array(corners)
    corners -= patch_size/2
    return coords+corners, np.array(accepted_flags)

        

    
    #patches = transform(patches.reshape(-1,2)).reshape(-1,4,2)
    #p = _extract_profile(img, patches[1,0], patches[1,1])

    #x = np.arange(p.shape[0])
    #res = np.polyfit(x, p, 4)
    #v = np.polyval(res, x)

    #plt.plot(x, p)
    #plt.plot(x, v)
    #plt.show()
    #exit()

    #for i in range(0,5):

    #    x, y, v = _extract_profile2d(img, patches[i], transform)


    #    xv, yv, vv = x.flatten(), y.flatten(), v.flatten()
    #    A = _bivariate_polynomial(xv, yv, 6)
    #    coeff, _, _, _ = np.linalg.lstsq(A, vv)

    #    rv = A.dot(coeff)

    #    m = np.argmin(rv)
    #    #print(xv[m], yv[m])

    #    plt.subplot(1,2,1)
    #    plt.pcolormesh(x, y, v, cmap=plt.get_cmap('gray'), vmin=v.min(), vmax=v.max())
    #    plt.plot([xv[m]], [yv[m]], marker='o', color='r')
    #    plt.subplot(1,2,2)
    #    plt.pcolormesh(x, y, rv.reshape(v.shape), cmap=plt.get_cmap('plasma'), vmin=v.min(), vmax=v.max())
    #    plt.show()
    #    plt.clf()
    

