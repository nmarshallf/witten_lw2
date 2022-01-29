import numpy as np
#import mrcfile
import scipy.ndimage as ndimage
import time
import matplotlib.pyplot as plt 
from scipy.io import loadmat 
from scipy.io import savemat 
from witten_lw2 import linear_w2, potential



def main():

    fontsize1 = 10
    fontsize2 = 14
    make_figures_1_and_2(fontsize1,fontsize2)
    make_figures_3_and_4(fontsize1,fontsize2)
    make_figure_5(fontsize1)
    plt.show()


def make_figures_1_and_2(fontsize1,fontsize2):

    

    # Parameters
    n = 2**7
    m = 2**6
    ts = np.array(range(n+1),dtype=np.float64)/n
    xs, ys = np.meshgrid(ts, ts, indexing='xy')
    dr = 0.05
    theta = 2*np.pi*np.array(range(m),dtype=np.float64)/m
    dxs = dr*np.cos(theta)
    dys = dr*np.sin(theta)

    dx = 0
    dy = 0
    X = func(xs,ys,dx,dy)
    dV = np.float64(1)/n**2
    Xint = np.sum(X*dV,axis=(0,1))
    X = X/Xint

    zz = zetabump(1.2*ts-.1,0.2,0.8)

    # Visualize data
    plt.figure() 
    color_map = plt.cm.get_cmap('gray').reversed()
    plt.imshow(X,cmap=color_map,extent=[0,1,0,1])
    plt.colorbar()
    plt.xticks(fontsize=fontsize1)
    plt.yticks(fontsize=fontsize1)
    plt.savefig('fig01a.eps', format='eps',bbox_inches='tight')


    f = X;
    sz = X.shape

    YS = np.zeros((n+1,n+1,m),dtype=np.float64)
    for i in range(m):
        dx = dxs[i]
        dy = dys[i]
        YS[:,:,i] = func(xs,ys,dx,dy)

    v = np.zeros(m,dtype=np.float64)
    q, ff = potential(f,eps1=1e-6,tau = 5e-3)
    szq = q.shape
    q = np.reshape(q,sz)

    # Visualize potential 
    plt.figure()
    plt.imshow(q,cmap=color_map,extent=[0,1,0,1])
    plt.colorbar()
    q = np.reshape(q,szq)
    plt.xticks(fontsize=fontsize1)
    plt.yticks(fontsize=fontsize1)
    plt.savefig('fig01b.eps', format='eps',bbox_inches='tight')

    for i in range(m):
        Y = YS[:,:,i]
        v[i] = linear_w2(X,Y,q,ff,verbose=True,tol=1e-4)

    u = np.zeros(m,dtype=np.float64)
    u1 = np.zeros(m,dtype=np.float64)
    f = np.ones(sz,dtype=np.float64)
    q, ff = potential(f)
    for i in range(m):
        Y = YS[:,:,i]
        u[i] = linear_w2(X,Y,q,ff)
    
    # Euclidean
    e = np.zeros(m,dtype=np.float64)
    for i in range(m):
        Y = YS[:,:,i]
        e[i] = l2dist(X,Y,sz)
    
    xxt = dxs
    yyt = dys
    xxt = np.concatenate((xxt,xxt[:1]),axis=0)
    yyt = np.concatenate((yyt,yyt[:1]),axis=0)
    tt = np.concatenate((theta,theta[:1]),axis=0)

    # Euclidean
    plt.figure()
    xx = e*np.cos(theta)
    yy = e*np.sin(theta)
    xx = np.concatenate((xx,xx[:1]),axis=0)
    yy = np.concatenate((yy,yy[:1]),axis=0)
    tt = np.concatenate((theta,theta[:1]),axis=0)
    plt.plot(xx,yy,'k')
    plt.plot(xxt,yyt,'k:')
    plt.axis('equal')
    plt.xticks(fontsize=fontsize2)
    plt.yticks(fontsize=fontsize2)
    plt.savefig('fig02a.eps', format='eps',bbox_inches='tight')


    # Unweighted Sobolev 
    plt.figure()
    xx = u*np.cos(theta)
    yy = u*np.sin(theta)
    xx = np.concatenate((xx,xx[:1]),axis=0)
    yy = np.concatenate((yy,yy[:1]),axis=0)
    tt = np.concatenate((theta,theta[:1]),axis=0)
    plt.plot(xx,yy,'k')
    plt.plot(xxt,yyt,'k:')
    plt.axis('equal')
    plt.xticks(fontsize=fontsize2)
    plt.yticks(fontsize=fontsize2)
    plt.savefig('fig02b.eps', format='eps',bbox_inches='tight')

    # Linearization of W2
    plt.figure()
    xx = v*np.cos(theta)
    yy = v*np.sin(theta)
    xx = np.concatenate((xx,xx[:1]),axis=0)
    yy = np.concatenate((yy,yy[:1]),axis=0)
    tt = np.concatenate((theta,theta[:1]),axis=0)
    plt.plot(xx,yy,'k')
    plt.plot(xxt,yyt,'k:')
    plt.axis('equal')
    plt.xticks(fontsize=fontsize2)
    plt.yticks(fontsize=fontsize2)
    plt.savefig('fig02c.eps', format='eps',bbox_inches='tight')



    return



def make_figures_3_and_4(fontsize1,fontsize2):

    # Parameters
    n = 2**7
    m = 2**6
    ts = np.array(range(n+1),dtype=np.float64)/n
    xs, ys = np.meshgrid(ts, ts, indexing='xy')
    dr = 0.01
    theta = 2*np.pi*np.array(range(m),dtype=np.float64)/m
    dxs = dr*np.cos(theta)
    dys = dr*np.sin(theta)

    dx = 0
    dy = 0
    dV = np.float64(1)/n**2
    X = gaussian_2d_example(n,dmx=0,dmy=0,dsx=0,dsy=0)
    Xint = np.sum(X*dV,axis=(0,1))
    X = X/Xint

    zz = zetabump(1.2*ts-.1,0.2,0.8)

    # Visualize data
    plt.figure() 
    color_map = plt.cm.get_cmap('gray').reversed()
    plt.imshow(X,cmap=color_map,extent=[0,1,0,1])
    plt.colorbar()
    plt.xticks(fontsize=fontsize1)
    plt.yticks(fontsize=fontsize1)
    plt.savefig('fig03a.eps', format='eps',bbox_inches='tight')


    f = X;
    sz = X.shape

    YS = np.zeros((n+1,n+1,m),dtype=np.float64)
    for i in range(m):
        dx = dxs[i]
        dy = dys[i]
        YS[:,:,i] = gaussian_2d_example(n,dmx=0,dmy=0,dsx=dx,dsy=dy)

    v = np.zeros(m,dtype=np.float64)
    q,ff = potential(f,eps1=1e-4,tau = 1e-3)
    szq = q.shape
    q = np.reshape(q,sz)
    plt.figure()
    plt.imshow(q,cmap=color_map,extent=[0,1,0,1])
    plt.colorbar()
    plt.xticks(fontsize=fontsize1)
    plt.yticks(fontsize=fontsize1)
    plt.savefig('fig03b.eps', format='eps',bbox_inches='tight')
    q = np.reshape(q,szq)

    for i in range(m):
        Y = YS[:,:,i]
        v[i] = linear_w2(X,Y,q,ff,verbose=True,tol=1e-4)

    u = np.zeros(m,dtype=np.float64)
    u1 = np.zeros(m,dtype=np.float64)
    #ratio = np.zeros(m,dtype=np.float64)
    f = np.ones(sz,dtype=np.float64)
    q,ff = potential(f)
    for i in range(m):
        Y = YS[:,:,i]
        u[i] = linear_w2(X,Y,q,ff,verbose=True)
   
    err = np.abs(v - dr)/np.abs(dr)

    
    #plt.figure()
    #plt.imshow(f)
    #plt.title('f')

    # Translations
    #plt.figure()
    xxt = dxs
    yyt = dys
    xxt = np.concatenate((xxt,xxt[:1]),axis=0)
    yyt = np.concatenate((yyt,yyt[:1]),axis=0)
    tt = np.concatenate((theta,theta[:1]),axis=0)
    #plt.plot(xxt,yyt)
    #plt.title('translations')

    # lw2
    plt.figure()
    xx = v*np.cos(theta)
    yy = v*np.sin(theta)
    xx = np.concatenate((xx,xx[:1]),axis=0)
    yy = np.concatenate((yy,yy[:1]),axis=0)
    tt = np.concatenate((theta,theta[:1]),axis=0)
    plt.plot(xx,yy,'k')
    plt.plot(xxt,yyt,'k:')
    #plt.title('lw2')
    plt.axis('equal')
    plt.xticks(fontsize=fontsize2)
    plt.yticks(fontsize=fontsize2)
    plt.savefig('fig04c.eps', format='eps',bbox_inches='tight')


    # Unweighted 
    plt.figure()
    xx = u*np.cos(theta)
    yy = u*np.sin(theta)
    xx = np.concatenate((xx,xx[:1]),axis=0)
    yy = np.concatenate((yy,yy[:1]),axis=0)
    tt = np.concatenate((theta,theta[:1]),axis=0)
    plt.plot(xx,yy,'k')
    plt.plot(xxt,yyt,'k:')
    #plt.title('unweighted')
    plt.axis('equal')
    plt.xticks(fontsize=fontsize2)
    plt.yticks(fontsize=fontsize2)
    plt.savefig('fig04b.eps', format='eps',bbox_inches='tight')

    # L2
    e = np.zeros(m,dtype=np.float64)
    for i in range(m):
        Y = YS[:,:,i]
        e[i] = l2dist(X,Y,sz)
    plt.figure()
    xx = e*np.cos(theta)
    yy = e*np.sin(theta)
    xx = np.concatenate((xx,xx[:1]),axis=0)
    yy = np.concatenate((yy,yy[:1]),axis=0)
    tt = np.concatenate((theta,theta[:1]),axis=0)
    plt.plot(xx,yy,'k')
    plt.plot(xxt,yyt,'k:')
    #plt.title('L2')
    plt.axis('equal')
    plt.xticks(fontsize=fontsize2)
    plt.yticks(fontsize=fontsize2)
    plt.savefig('fig04a.eps', format='eps',bbox_inches='tight')






def func(xs,ys,dx,dy):

    zeta = zetabump(1.2*(xs+dx)-.1,0.2,0.8)*zetabump(1.2*(ys+dy)-.1,0.2,0.8)
    v = np.exp(8*(xs+dx))*zeta*(np.cos(16*np.pi*(xs+dx))+1) 
           # *(np.sin(np.pi*(ys+dy))+1)/4

    return v



def zetabump(r,a,b):
   
    sz = r.shape 
    id0 = r <= a
    id = (r>a)*(r<b)
    id1 = r >= b
    
    v = np.zeros(sz,np.float64)
    v[id0] = zeta(r[id0]/a)
    v[id] = np.ones(v[id].shape,dtype=np.float64)
    v[id1] = zeta((1-r[id1])/(1-b))

    return v


def l2dist(X,Y,sz):

    # Definitions
    n = sz[0] - 1
    half = np.float64(1)/2
    sqrt2 = np.sqrt(np.float64(2))

    # volume element
    dV = np.float64(1)/n**2

    X = weight_endpoints(X,half,sz)
    Y = weight_endpoints(Y,half,sz)

    Xint = np.sum(X*dV,axis=(0,1))
    Yint = np.sum(Y*dV,axis=(0,1))
    X = X/Xint
    Y = Y/Yint

    u = X - Y

    u = weight_endpoints(u,sqrt2,sz)
 
    v = np.sum(u**2,axis=(0,1))*dV
    v = np.sqrt(v)

    return v




def zeta(r):

    sz = r.shape
    id0 = r<=0
    id = (r>0)*(r<1)
    id1 = r>=1
    v = np.zeros(sz,np.float64)
    
    v[id1] = 1
    h0 = np.exp(-1/r[id])
    h1 = np.exp(-1/(1-r[id]))
    v[id] = h0/(h0 + h1)

    ### TEST CODE
    ##plt.figure()
    ##plt.plot(r,v)
    return v



def weight_endpoints(x,w,sz):
    szx = x.shape
    n = sz[0] - 1
    ia = np.array(range(n+1))
    ie = np.array([0,n])
    idx = np.ix_(ia,ie)
    idy = np.ix_(ie,ia)
    x = np.reshape(x,sz)
    x[idx] = x[idx]*w
    x[idy] = x[idy]*w
    return x

def gaussian_2d_example(n,dmx=0.01,dmy=0.02,dsx=0.03,dsy=0.04):

    x = np.pi*np.float64(range(n+1))/n
    y = np.pi*np.float64(range(n+1))/n
    xs, ys = np.meshgrid(x, y, indexing='ij')

    # Parameters
    mx = np.float64(np.pi)/2
    sx = np.float64(np.pi)/16
    mx1 = mx + dmx*np.pi
    sx1 = sx + dsx*np.pi

    my = np.float64(np.pi)/2
    sy = np.float64(np.pi)/16
    my1 = my + dmy*np.pi
    sy1 = sy + dsy*np.pi
    
    # Function Handles
    fx = lambda x: 1/(np.sqrt(2*np.pi)*sx)*np.exp(-(x-mx)**2/(2*sx**2))
    hx = lambda x: 1/(np.sqrt(2*np.pi)*sx1)*np.exp(-(x-mx1)**2/(2*sx1**2))
    fxp = lambda x: -(x-mx)/sx**2*fx(x)
    fxpp = lambda x: -1/sx**2*fx(x) + (x-mx)**2/sx**4*fx(x)

    fy = lambda y: 1/(np.sqrt(2*np.pi)*sy)*np.exp(-(y-my)**2/(2*sy**2))
    hy = lambda y: 1/(np.sqrt(2*np.pi)*sy1)*np.exp(-(y-my1)**2/(2*sy1**2))
    fyp = lambda y: -(y-my)/sy**2*fy(y)
    fypp = lambda y: -1/sy**2*fy(y) + (y-my)**2/sy**4*fy(y)

    # Evaluate
    fs = fx(xs)*fy(ys)
    hs = hx(xs)*hy(ys)
    fgs = np.sqrt((fy(ys)*fxp(xs))**2 + (fx(xs)*fyp(ys))**2)
    fds = fy(ys)*fxpp(xs) + fy(xs)*fypp(ys)
    gcheck = -(1/4)*fs**(-2)*fgs**2 + (1/2)*fs**(-1)*fds
  
    sol = np.sqrt(dmx**2+dmy**2+dsx**2+dsy**2)
    sol = sol/(np.pi)

    return hs


def make_figure_5(fontsize1):

    # Parameters
    n = 2**7
    m = 2**7
    ts = np.array(range(n+1),dtype=np.float64)/n
    xs, ys = np.meshgrid(ts, ts, indexing='xy')
    dr = 0.05
    theta = 2*np.pi*np.array(range(m),dtype=np.float64)/m
    dxs = dr*np.cos(theta)
    dys = dr*np.sin(theta)

    dx = 0
    dy = 0
    X = jellyshift(dx,dy)
    dV = np.float64(1)/n**2
    Xint = np.sum(X*dV,axis=(0,1))
    X = X/Xint

    zz = zetabump(1.2*ts-.1,0.2,0.8)
    #plt.figure()
    #plt.plot(ts,zz,'k-')

    # Visualize data
    plt.figure() 
    color_map = plt.cm.get_cmap('gray').reversed()
    plt.imshow(X,cmap=color_map,extent=[0,1,0,1])
    plt.colorbar()
    plt.xticks(fontsize=fontsize1)
    plt.yticks(fontsize=fontsize1)
    plt.savefig('fig05a.eps', format='eps',bbox_inches='tight')


    f = X;
    sz = X.shape

    YS = np.zeros((n+1,n+1,m),dtype=np.float64)
    for i in range(m):
        dx = dxs[i]
        dy = dys[i]
        YS[:,:,i] = jellyshift(dx,dy)
        #plt.figure()
        #plt.imshow(YS[:,:,i])


    v = np.zeros(m,dtype=np.float64)
    tau = 5e-3
    q, ff = potential(f,eps1=1e-2,tau = tau)
    szq = q.shape
    q = np.reshape(q,sz)
    plt.figure()
    plt.imshow(q,cmap=color_map,extent=[0,1,0,1])
    plt.colorbar()
    plt.xticks(fontsize=fontsize1)
    plt.yticks(fontsize=fontsize1)
    plt.savefig('fig05b.eps', format='eps',bbox_inches='tight')
    q = np.reshape(q,szq)

    f = X;
    sz = X.shape

    YS = np.zeros((n+1,n+1,m),dtype=np.float64)
    for i in range(m):
        dx = dxs[i]
        dy = dys[i]
        YS[:,:,i] = jellyshift(dx,dy)


    v = np.zeros(m,dtype=np.float64)
    tau = 5e-3
    q, ff = potential(f,eps1=1e-2,tau = tau)
    t0 = time.time()
    for i in range(m):
        Y = YS[:,:,i]
        v[i] = linear_w2(X,Y,q,ff,verbose=True)
    t1 = time.time()
    print("Average time=",(t1-t0)/m)
    print("number=",m)


    return


def jellyshift(dx,dy):

    ns = 108
    nd = 2**7 + 1
    nc = nd//2
    dn = (nd-ns)//2
    
    #with mrcfile.open('FakeKvMapAlphaOne.mrc') as mrc_a1:
    #    a1 = np.maximum(mrc_a1.data,0)
    #with mrcfile.open('FakeKvMapAlphaTwo.mrc') as mrc_a2:
    #    a2 = np.maximum(mrc_a2.data,0)
    #with mrcfile.open('FakeKvMapBeta.mrc') as mrc_b:
    #    b = np.maximum(mrc_b.data,0)
    #vol = a1 + a2 + b;

    #mdic = {"vol": vol}
    #savemat("data.mat", mdic)
    data = loadmat("data.mat")
    vol = data["vol"]
    
    
    dx = dx*(nd-1)  
    dy = dy*(nd-1)  
    image = np.zeros((nd,nd),dtype=np.float64)
    image[dn:ns+dn,dn:ns+dn] = np.sum(vol,axis=1)
    image = ndimage.shift(image,np.array([dx,dy]))
    image = np.maximum(image,0)

    return image





if __name__ == "__main__":
    main()

