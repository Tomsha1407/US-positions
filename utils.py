import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
import scipy
import scipy.ndimage
import scipy.io
import torch.nn as nn
import torchvision.transforms as T

import DAS
import Phantom


class Grid:
    def __init__(self, Lx, Lz, nx, nz, nt, dt):
        # Space
        self.Lx = Lx
        self.Lz = Lz
        self.nx = nx
        self.nz = nz
        self.dx = self.Lx / self.nx
        self.dz = self.Lz / self.nz

        # Time
        self.nt = nt
        self.CFL = 0.5

        self.device = torch.device("cuda")
        self.dtype = torch.float64

        # Number of operation (same as the size of the kernel)
        self.nop = 5

        self.PML_size = 20
        self.PML_max_damping = 0.1

        self.plotDas = False
        self.stdNoise = 0.0003
        self.factor = 0.95

        self.reconstructVelocity = True
        self.reconstructDensity = True
        self.reconstructDamping = True
        self.reconstructBeta = False
        self.reconstructLoc = True

        self.mask = torch.tensor(1.0, dtype=self.dtype, device=self.device)
        self.dt = torch.tensor(dt, dtype=self.dtype, device=self.device)
        self.T = self.dt.detach().cpu().numpy() * self.nt


class Prob:
    def __init__(self, elementsLocations, elementsDelays, elementsApodizations, basePulse, numChannels, c0, f0, grid):
        # The number of channels in the probe
        self.numChannels = 5# numChannels

        # The location of the elements in the probe. An numChannels x 2 matrix
        # elementsLocations_tmp = (np.round_(elementsLocations * (c0 / f0) / grid.dx) + int(grid.nx / 2)).astype(np.int)
        # self.elementsLocations = np.zeros(elementsLocations.shape).astype(np.int)
        # self.elementsLocations[:, 0], self.elementsLocations[:, 1] = elementsLocations_tmp[:, 1] - int(grid.nx / 2) + grid.PML_size + 1, elementsLocations_tmp[:, 0]
        self.elementsDelays = elementsDelays
        self.elementsApodizations = elementsApodizations

        # self.elementsLocations = self.elementsLocations[int((numChannels - self.numChannels) / 2):int((numChannels + self.numChannels) / 2), :]
        self.elementsDelays = self.elementsDelays[:, int((numChannels - self.numChannels) / 2):int((numChannels + self.numChannels) / 2)]
        self.elementsApodizations = self.elementsApodizations[:, int((numChannels - self.numChannels) / 2):int((numChannels + self.numChannels) / 2)]

        # elementsPhysicalLocations_tmp = elementsLocations * (c0 / f0)
        # self.elementsPhysicalLocations = np.zeros(elementsLocations.shape)
        # self.elementsPhysicalLocations[:, 0], self.elementsPhysicalLocations[:, 1] = elementsPhysicalLocations_tmp[:, 1] - (grid.PML_size + 1) * grid.dx + grid.Lx / 2, elementsPhysicalLocations_tmp[:, 0]
        # self.elementsPhysicalLocations = self.elementsPhysicalLocations[int((numChannels - self.numChannels) / 2):int((numChannels + self.numChannels) / 2), :]

        # The dominant frequency of the elements (Hz)
        self.f0 = f0

        # Lateral properties of the prob
        self.numLaterals = self.numChannels

        self.short_basePulse = basePulse
        self.base_pulse = torch.zeros(grid.nt)
        timeOffset = 0
        self.base_pulse[timeOffset:(timeOffset + len(basePulse))] = torch.Tensor(basePulse)


class PhysicalModel:
    def __init__(self, c0):
        self.c0 = c0
        self.density = 1


        self.vmax_velocity = c0 + 55e-6
        self.vmin_velocity = c0 - 55e-6
        self.vmin_density = 0.8
        self.vmax_density = 1.2
        self.vmin_damping = -1.0
        self.vmax_damping =  1.0
        self.vmin_beta = nlp2beta('water') - 5
        self.vmax_beta = nlp2beta('water') + 5
        self.vmax_location = 255  ## need to adjust!!!!!!1
        self.vmin_location = 0  ## need to adjust!!!!!
        # self.vmax_location[0] = 255  ## need to adjust!!!!!
        # self.vmax_location[1] = 255  ## need to adjust!!!!!!1
        # self.vmin_location[0] = 0  ## need to adjust!!!!!
        # self.vmin_location[1] = 0  ## need to adjust!!!!!!

class Properties:
    def __init__(self):
        pass


def circleMask(x0, z0, Sigma, amplitude, grid):
    """
    Define an ellipse centered at x0 and z0, with a value "amplitude".
    :param x0: x coordinate of the center
    :param z0: z coordinate of the center
    :param Sigma: The ellipse matrix
    :param amplitude: The amplitude of the mask
    :param grid
    :return: A mask with value "amplitude" inside the ellipse, and 0 outside of it.
    """
    z, x = np.ogrid[-grid.nz / 2: grid.nz / 2, -grid.nx / 2: grid.nx / 2]
    x = torch.from_numpy(x).to(grid.dtype).to(grid.device)
    z = torch.from_numpy(z).to(grid.dtype).to(grid.device)
    SigmaInv = np.linalg.inv(np.array(Sigma) @ np.array(Sigma))
    mask = ((x - x0) ** 2 * SigmaInv[1, 1] + (z - z0) ** 2 * SigmaInv[0, 0] \
           + (x - x0) * (z - z0) * (SigmaInv[0, 1] + SigmaInv[1, 0])) <= 1
    return amplitude * mask


def getGTProperties(grid, prob, physicalModel):
    """
    :return: The GT properties of the material.
    """
    SigmaLiver, SigmaFat, liverLocation, fatLocation = getMasks()

    # SigmaPin = [[1.0, 0.0],
    #             [0.0,  1.0]]
    # pin_x, pin_z = 0, 0
    GTVelocities = torch.zeros([grid.nz, grid.nx], dtype=grid.dtype, device=grid.device) + physicalModel.c0
    # GTVelocities += circleMask(x0=pin_x, z0=pin_z, Sigma=SigmaPin, amplitude=100e-6, grid=grid)
    # GTVelocities = torch.from_numpy(scipy.ndimage.gaussian_filter(GTVelocities.to(dtype=torch.float32).detach().cpu().numpy(), sigma=1.0)).to(grid.dtype).to(grid.device)

    # GTDensities = torch.ones([grid.nz, grid.nx], dtype=grid.dtype, device=grid.device)
    # GTDensities += circleMask(x0=pin_x, z0=pin_z, Sigma=SigmaPin, amplitude=0.2, grid=grid)
    # GTDensities = torch.from_numpy(scipy.ndimage.gaussian_filter(GTDensities.to(dtype=torch.float32).detach().cpu().numpy(), sigma=1.0)).to(grid.dtype).to(grid.device)

    maskT1 = circleMask(x0=liverLocation[0], z0=liverLocation[1], Sigma=SigmaLiver, amplitude=170e-6,
                        grid=grid)  # circleMask(x0=25, z0=-grid.nz / 2 + 105, r=15, amplitude=-50, grid=grid)
    # maskT2 = circleMask(x0=fatLocation[0], z0=fatLocation[1], Sigma=SigmaFat, amplitude=30e6,
    #                     grid=grid)  # circleMask(x0=-25, z0=-grid.nz / 2 + 80, r=20, amplitude=50, grid=grid)
    GTVelocities += maskT1
    # GTVelocities = torch.from_numpy(scipy.ndimage.gaussian_filter(GTVelocities.to(dtype=torch.float32).detach().cpu().numpy(), sigma=0.5)).to(grid.dtype).to(grid.device)

    GTDensities = physicalModel.density*torch.ones([grid.nz, grid.nx], dtype=grid.dtype, device=grid.device)
    GTDensities += circleMask(x0=liverLocation[0], z0=liverLocation[1], Sigma=SigmaLiver, amplitude=0.06, grid=grid)
    # GTDensities += circleMask(x0=fatLocation[0], z0=fatLocation[1], Sigma=SigmaFat, amplitude=-0.05, grid=grid)
    #GTDensities = torch.from_numpy(scipy.ndimage.gaussian_filter(GTDensities.to(dtype=torch.float32).detach().cpu().numpy(), sigma=0.2)).to(grid.dtype).to(grid.device)

    GTDamping = initializeDamping(grid, prob, physicalModel)
    # GTDamping += circleMask(x0=pin_x, z0=pin_z, Sigma=SigmaPin, amplitude=0.0, grid=grid)
    # GTDamping += circleMask(x0=fatLocation[0], z0=fatLocation[1], Sigma=SigmaFat, amplitude=tissueDamping('fat', grid, prob) - tissueDamping('water', grid, prob), grid=grid)
    GTDamping += circleMask(x0=liverLocation[0], z0=liverLocation[1], Sigma=SigmaLiver, amplitude=tissueDamping('liver', grid, prob) - tissueDamping('water', grid, prob), grid=grid)


    GTBeta = 0* nlp2beta('water') * torch.ones([grid.nz, grid.nx], dtype=grid.dtype, device=grid.device)
    # GTBeta += circleMask(x0=liverLocation[0], z0=liverLocation[1], Sigma=SigmaLiver, amplitude=nlp2beta('liver') - nlp2beta('water'), grid=grid)
    # GTBeta += circleMask(x0=fatLocation[0], z0=fatLocation[1], Sigma=SigmaFat, amplitude=nlp2beta('fat') - nlp2beta('water'), grid=grid)

    a_cir = 0
    b_cir = grid.PML_size +1
    c_cir = 3
    #GTLoc = parbula_location(prob.numChannels,a_cir,b_cir,c_cir,grid)
    GTLoc = linear_location(prob.numChannels,a_cir,b_cir,grid)

    properties = Properties()
    properties.velocity = GTVelocities
    properties.density = GTDensities
    properties.damping = GTDamping
    properties.beta = GTBeta
    properties.loc = GTLoc
    return properties


def getMasks():
    SigmaLiver = [[4.0, 0.0],
                  [0.0,  4.0]]
    SigmaFat   = [[8.0, 0.0],
                  [0.0,  8.0]]
    liverLocation = [0, 0]
    fatLocation = [-15, -16]
    return SigmaLiver, SigmaFat, liverLocation, fatLocation


def getRefProperties(grid, prob, physicalModel):
    """
    :return: The reference properties.
    """
    properties = Properties()
    properties.velocity = torch.zeros([grid.nz, grid.nx], dtype=grid.dtype, device=grid.device) + physicalModel.c0
    properties.density = torch.ones([grid.nz, grid.nx], dtype=grid.dtype, device=grid.device)
    properties.damping = initializeDamping(grid, prob, physicalModel)
    properties.beta = 5 * torch.zeros([grid.nz, grid.nx], dtype=grid.dtype, device=grid.device)
    a_cir = 5
    b_cir = 3
    c_cir = 3
    # properties.loc = parbula_location(prob.numChannels,a_cir,b_cir,c_cir,grid)  #????????
    properties.loc = linear_location(prob.numChannels,a_cir,b_cir,grid)


    return properties


def initializeProperties(trans, transRef, GTProperties, grid, prob, physicalModel, DASInitialization):
    """
    Initialize the properties.
    The velocity can be initialized using the corresponding DAS.
    :param trans: The channel data obtain. Will be used to compute the DAS image.
    :param transRef: The channel data obtain from the reference material. Will be used to compute the DAS image.
    :param GTProperties: The GT properties of the material.
    :param DASInitialization: True if initialize with DAS.
    :return: The initialization properties.
    """
    initialVelocities = torch.zeros([grid.nz, grid.nx], dtype=grid.dtype, device=grid.device) + physicalModel.c0 #+ 10 * (torch.rand([grid.nz, grid.nx], dtype=grid.dtype, device=grid.device) - 0.5)

    initialDensities = physicalModel.density * torch.ones([grid.nz, grid.nx], dtype=grid.dtype, device=grid.device) #+ 0.01 * torch.randn([grid.nz, grid.nx], dtype=grid.dtype, device=grid.device)

    # if not grid.reconstructDamping:
    #     initialDamping = GTProperties.damping.clone()
    # else:
    initialDamping = initializeDamping(grid, prob, physicalModel)
    #
    # if not grid.reconstructBeta:
    #     initialBeta = GTProperties.beta.clone()
    # else:
    initialBeta = 0 * nlp2beta('water') * torch.ones([grid.nz, grid.nx], dtype=grid.dtype, device=grid.device)

    a_cir = 0  # i added
    b_cir = grid.PML_size  # i added
    c_cir = 5
    # initialLoc = parbula_location(prob.numChannels,a_cir,b_cir,c_cir,grid)  # i added
    initialLoc = linear_location(prob.numChannels, a_cir, b_cir, grid)

    properties = Properties()
    properties.velocity = initialVelocities
    properties.density = initialDensities
    properties.damping = initialDamping
    properties.beta = initialBeta
    properties.loc = initialLoc #i added

    return properties


def initializeDamping(grid, prob, physicalModel):
    """
    Initialize the damping, including the PML layer.
    :return: The initialized damping/attenuation.
    """
    damping = tissueDamping('water', grid, prob) * torch.ones([grid.nz, grid.nx], dtype=grid.dtype, device=grid.device)
    line_damping = ((torch.arange(0, grid.PML_size) / grid.PML_size).pow(2) * grid.PML_max_damping).to(grid.dtype).to(grid.device)
    line_damping = torch.tile(line_damping, (grid.nz, 1))
    damping[:grid.PML_size, :] += line_damping.flip(0, 1).transpose(0, 1)
    damping[-grid.PML_size:, :] += line_damping.transpose(0, 1)
    damping[:, :grid.PML_size] += line_damping.flip(0, 1)
    damping[:, -grid.PML_size:] += line_damping
    return damping


def plotProperties(propertiesPred, GTProperties, plotLoss, grid, prob, physicalModel):
    addTitle = True
    if plotLoss or grid.plotDas:
        fig, axes = plt.subplots(2, 6)
    else:
        fig, axes = plt.subplots(2, 4)
    plt.ion()
    fig.set_figheight(10)
    fig.set_figwidth(20)
    if addTitle: fig.suptitle('Backpropogating to the velocities')
    im = []

    #################### plot location ####################
    x = []
    for i in range(GTProperties.loc.detach().cpu().numpy().shape[0]):
        x.append(GTProperties.loc.detach().cpu().numpy()[i][0])

    y = []
    for i in range(GTProperties.loc.detach().cpu().numpy().shape[0]):
        y.append(GTProperties.loc.detach().cpu().numpy()[i][1])

    x_pred = []
    for i in range(propertiesPred.loc.detach().cpu().numpy().shape[0]):
        x_pred.append(propertiesPred.loc.detach().cpu().numpy()[i][0])

    y_pred = []
    for i in range(propertiesPred.loc.detach().cpu().numpy().shape[0]):
        y_pred.append(propertiesPred.loc.detach().cpu().numpy()[i][1])

    if addTitle: axes[0, 0].set_title('GT Velocities')
    img = axes[0, 0].imshow(GTProperties.velocity.detach().cpu(), interpolation='nearest', vmin=physicalModel.vmin_velocity, vmax=physicalModel.vmax_velocity, animated=True, cmap=plt.cm.RdBu)
    ln, = axes[0, 0].plot(y, x, 'ro')
    img.axes.add_image(ln)
    if not plotLoss:
        cbar = fig.colorbar(img, ax=axes[0, 0])
        if addTitle: cbar.ax.set_ylabel('[m / s]', rotation=270)
        if not addTitle: cbar.set_ticks([])

    if addTitle: axes[1, 0].set_title('Estimated Velocities')
    # img = axes[1, 0].imshow(propertiesPred.velocity.detach().cpu().numpy(), interpolation='nearest',
    #                         vmin=physicalModel.vmin_velocity, vmax=physicalModel.vmax_velocity, animated=True,
    #                         cmap=plt.cm.RdBu)
    # ln, = axes[1, 0].plot(y_pred,x_pred, 'ro')
    # img.axes.add_image(ln)
    im.append(axes[1, 0].imshow(propertiesPred.velocity.detach().cpu().numpy(), interpolation='nearest', vmin=physicalModel.vmin_velocity, vmax=physicalModel.vmax_velocity, animated=True, cmap=plt.cm.RdBu))

    im.append(axes[1, 0].plot(y_pred,x_pred,'ro')[0])
    if not plotLoss:
        cbar = fig.colorbar(im[-1], ax=axes[1, 0])
        if addTitle: cbar.ax.set_ylabel('[m / s]', rotation=270)
        if not addTitle: cbar.set_ticks([])

    if addTitle: axes[0, 1].set_title('GT Densities')
    img = axes[0, 1].imshow(GTProperties.density.detach().cpu(), interpolation='nearest', vmin=physicalModel.vmin_density, vmax=physicalModel.vmax_density, animated=True, cmap=plt.cm.RdBu)
    if not plotLoss:
        cbar = fig.colorbar(img, ax=axes[0, 1])
        if addTitle: cbar.ax.set_ylabel('[1000 * kg / m^3]', rotation=270)
        if not addTitle: cbar.set_ticks([])

    if addTitle: axes[1, 1].set_title('Estimated Densities')
    im.append(axes[1, 1].imshow(propertiesPred.density.detach().cpu().numpy(), interpolation='nearest', vmin=physicalModel.vmin_density, vmax=physicalModel.vmax_density, animated=True, cmap=plt.cm.RdBu))
    if not plotLoss:
        cbar = fig.colorbar(im[-1], ax=axes[1, 1])
        if addTitle: cbar.ax.set_ylabel('[1000 * kg / m^3]', rotation=270)
        if not addTitle: cbar.set_ticks([])

    if addTitle: axes[0, 2].set_title('GT Damping')
    img = axes[0, 2].imshow(GTProperties.damping.detach().cpu(), interpolation='nearest', vmin=physicalModel.vmin_damping, vmax=physicalModel.vmax_damping, animated=True, cmap=plt.cm.RdBu)
    if not plotLoss:
        cbar = fig.colorbar(img, ax=axes[0, 2])
        if addTitle: cbar.ax.set_ylabel('[1 / s]', rotation=270)
        if not addTitle: cbar.set_ticks([])

    if addTitle: axes[1, 2].set_title('Estimated Damping')
    im.append(axes[1, 2].imshow(propertiesPred.damping.detach().cpu().numpy(), interpolation='nearest', vmin=physicalModel.vmin_damping, vmax=physicalModel.vmax_damping, animated=True, cmap=plt.cm.RdBu))
    if not plotLoss:
        cbar = fig.colorbar(im[-1], ax=axes[1, 2])
        if addTitle: cbar.ax.set_ylabel('[1 / s]', rotation=270)
        if not addTitle: cbar.set_ticks([])

    if addTitle: axes[0, 3].set_title('GT nonlinearity')
    img = axes[0, 3].imshow(GTProperties.beta.detach().cpu(), interpolation='nearest', vmin=physicalModel.vmin_beta, vmax=physicalModel.vmax_beta, animated=True, cmap=plt.cm.RdBu)
    if not plotLoss:
        cbar = fig.colorbar(img, ax=axes[0, 3])
        if addTitle: cbar.ax.set_ylabel('', rotation=270)
        if not addTitle: cbar.set_ticks([])

    if addTitle: axes[1, 3].set_title('Estimated nonlinearity')
    im.append(axes[1, 3].imshow(propertiesPred.beta.detach().cpu().numpy(), interpolation='nearest', vmin=physicalModel.vmin_beta, vmax=physicalModel.vmax_beta, animated=True, cmap=plt.cm.RdBu))
    if not plotLoss:
        cbar = fig.colorbar(im[-1], ax=axes[1, 3])
        if addTitle: cbar.ax.set_ylabel('', rotation=270)
        if not addTitle: cbar.set_ticks([])

#################### plot location ####################
    # x = []
    # for i in range(GTProperties.loc.detach().cpu().numpy().shape[0]):
    #     x.append(GTProperties.loc.detach().cpu().numpy()[i][0])
    #
    # y = []
    # for i in range(GTProperties.loc.detach().cpu().numpy().shape[0]):
    #     y.append(GTProperties.loc.detach().cpu().numpy()[i][1])

    if addTitle: axes[0, 4].set_title('GT Location')
    ln, = axes[0, 4].plot(y,x,'ro')
    im.append(ln)
    #im.append(axes[0, 4].imshow(x,y, interpolation='nearest', vmin=physicalModel.vmin_location, vmax=physicalModel.vmax_location, animated=True, cmap='gray'))
    if not plotLoss:
        cbar = fig.colorbar(img, ax=axes[0, 4])
        if addTitle: cbar.ax.set_ylabel('', rotation=270)
        if not addTitle: cbar.set_ticks([])

    # x_pred= []
    # for i in range(propertiesPred.loc.detach().cpu().numpy().shape[0]):
    #     x_pred.append(propertiesPred.loc.detach().cpu().numpy()[i][0])
    #
    # y_pred = []
    # for i in range(propertiesPred.loc.detach().cpu().numpy().shape[0]):
    #     y_pred.append(propertiesPred.loc.detach().cpu().numpy()[i][1])
    if addTitle: axes[1, 4].set_title('Estimated location')
    ln2, = axes[1, 4].plot(y_pred,x_pred,'ro')
    im.append(ln2)
    # im.append(axes[1, 4].imshow(propertiesPred.loc.detach().cpu().numpy(), interpolation='nearest', vmin=physicalModel.vmin_location, vmax=physicalModel.vmax_location, animated=True, cmap='gray'))
    if not plotLoss:
        cbar = fig.colorbar(im[-1], ax=axes[1, 4])
        if addTitle: cbar.ax.set_ylabel('', rotation=270)
        if not addTitle: cbar.set_ticks([])


    if grid.plotDas:
        axes[1, 4].set_title('B-mode image')
        im.append(axes[1, 4].imshow(grid.das.detach().cpu().numpy(), interpolation='nearest', animated=True, cmap=plt.cm.binary))
        vmax_pressure = 15000 / grid.dt.detach()**2 * max([np.abs(prob.pulses.min().cpu()), np.abs(prob.pulses.max().cpu())])
        axes[0, 4].set_title('Signal on transducer')
        im.append(axes[0, 4].imshow(grid.channelData.detach().cpu().numpy(), interpolation='nearest', animated=True, vmin=-vmax_pressure, vmax=vmax_pressure, cmap=plt.cm.binary))
        cbar = fig.colorbar(im[-1], ax=axes[0, 4])
        cbar.ax.set_ylabel('Pressure [Pa]', rotation=270)

    for i in range(2):
        for j in range(5):
            axes[i, j].get_xaxis().set_visible(False)
            axes[i, j].get_yaxis().set_visible(False)

    fig.show()
    fig.canvas.draw()
    fig.canvas.flush_events()
    return fig, axes, im


def plotChannelData(channelData, fig, axes, vmax_pressure):
    plt.ion()
    fig.suptitle('Signal on transducer')
    axes.imshow(channelData.detach().cpu().numpy(), interpolation='nearest', animated=True, vmin=-vmax_pressure, vmax=vmax_pressure, cmap=plt.cm.binary)
    fig.show()
    fig.canvas.draw()
    fig.canvas.flush_events()


def initializeSimulator(f0):
    # Initializing the geometry
    elementsLocations, elementsDelays, elementsApodizations, basePulse, numChannels, nt, dt_phantom, dx_lambda, nx, L, trans = Phantom.loadMatData()

    L = 0.02 # in meters
    oversampling_time = 1 # For the moment, needs to be 1
    oversampling_space = 2


    dt = dt_phantom / oversampling_time
    c0 = 1540.0e-6 #1480e-6 #1540.0e-6  # velocity m / microsec (can be an array) - 580
    dx = c0 / f0 * dx_lambda / oversampling_space
    nt = int( np.ceil( 2 * L / (c0 * dt) ))
    nx = int(np.ceil(L / dx))
    print('nx =', nx, ', nt =', nt)
    trans = trans[:nt, :, :]

    grid = Grid(Lx=L, Lz=L, nx=nx, nz=nx, nt=nt, dt=dt)
    grid.trans = torch.from_numpy(trans).to(dtype=grid.dtype).to(device=grid.device)
    grid.oversampling_time = oversampling_time
    grid.distance_elements = c0 / f0 * dx_lambda
    grid.NLA_model = True

    print('dx =', grid.dx * 1e3, '[mm]')

    # Initializing the prob and the physical model
    prob = Prob(elementsLocations, elementsDelays, elementsApodizations, basePulse, numChannels=numChannels, c0=c0, f0=f0, grid=grid)

    physicalModel = PhysicalModel(c0=c0)
    return grid, prob, physicalModel


def saveProperties(name, properties, dir='Results'):
    np.save(dir + '/velocityPred' + name, properties.velocity.detach().cpu().numpy())
    np.save(dir + '/densityPred' + name, properties.density.detach().cpu().numpy())
    np.save(dir + '/dampingPred' + name, properties.damping.detach().cpu().numpy())
    np.save(dir + '/betaPred' + name, properties.beta.detach().cpu().numpy())
    np.save(dir + '/locPred' + name, properties.loc.detach().cpu().numpy())



def nlp2beta(tissue):
    """
    Nonlinear parameter to beta.
    :param tissue: The name of the tissue.
    :return: The nonlinearity, beta, of the tissue.
    """
    NonLinearParam = {'water': 5.2, 'fat': 10, 'liver': 6.8}
    return 1 + NonLinearParam[tissue] / 2


def tissueDamping(tissue, grid, prob):
    """
    Attenuation of the tissue.
    :param tissue: The name of the tissue.
    :return: The attenuation of the tissue.
    """
    a = {'water': 0.002, 'fat': 0.6, 'liver': 0.9}
    b = {'water': 2.0  , 'fat': 1.0, 'liver': 1.1}
    return a[tissue] * (prob.f0) ** b[tissue] * grid.dt


def getNoise(grid, prob):
    """
    Define the noise in the channel data.
    :return: Return the i.i.d. normal noise.
    """
    mean = torch.zeros([grid.nt, prob.numChannels], dtype=grid.dtype, device=grid.device)
    std = grid.stdNoise * torch.ones([grid.nt, prob.numChannels], dtype=grid.dtype, device=grid.device)
    noise = []
    [noise.append(torch.normal(mean=mean, std=std).to(grid.device).to(grid.dtype).detach()) for _ in range(prob.numLaterals)]
    return noise


def RMSE(x, y):
    """
    Compute the normalized RMSE score.
    :param x: First tensor
    :param y: Second tensor
    :return: NRMSE
    """
    mseLoss = nn.MSELoss()
    RMSE = mseLoss(x, y).sqrt()
    NRMSE = RMSE / (x.max() - x.min())
    return NRMSE.item()


def CNR(x, maskObject, maskBackground):
    """
    Compute the CNR according to two regions: object and background.
    :param x: The signal.
    :param maskObject: Region of the object.
    :param maskBackground: Region of the background.
    :return: The CNR.
    """
    muObject = x[maskObject].mean()
    varObject = x[maskObject].std() ** 2
    muBg = x[maskBackground].mean()
    varBg = x[maskBackground].std() ** 2
    CNR = 2 * (muObject - muBg) ** 2 / (varObject + varBg)
    return CNR


def SNR(signal, std):
    """
    SNR.
    :param signal: The signal.
    :param std: The std of the normal noise.
    :return: The SNR.
    """
    ES2 = (signal - signal.mean()).pow(2).mean()
    return 10 * torch.log10(ES2 / (std**2)).item()


def dict2prop(dict):
    properties = Properties()
    properties.velocity = dict['velocity']
    properties.density = dict['density']
    properties.damping = dict['damping']
    properties.beta = dict['beta']
    properties.loc = dict['location']
    return properties


def prop2dict(prop):
    dict = {'velocity': prop.velocity, 'density': prop.density, 'damping': prop.damping, 'beta': prop.beta, 'location':prop.loc}
    return dict

def loadProperties(name, dir='Results'):
    properties = Properties()
    properties.velocity = torch.from_numpy(np.load(dir + '/velocityPred' + name + '.npy', allow_pickle=True))
    properties.density = torch.from_numpy(np.load(dir + '/densityPred' + name + '.npy', allow_pickle=True))
    properties.damping = torch.from_numpy(np.load(dir + '/dampingPred' + name + '.npy', allow_pickle=True))
    properties.beta = torch.from_numpy(np.load(dir + '/betaPred' + name + '.npy', allow_pickle=True))
    properties.loc = torch.from_numpy(np.load(dir + '/locPred' + name + '.npy', allow_pickle=True))
    return properties


def applyDelaysPulse(lateral, channel, grid, prob):
    basePulse = prob.base_pulse
    delay = prob.elementsDelays[lateral, channel] * 1e-6
    discreteDelay = int(delay / grid.dt.detach().cpu().numpy())
    delayedPulse = torch.zeros(basePulse.shape)
    if discreteDelay == 0:
        delayedPulse = basePulse.clone()
    else:
        delayedPulse[discreteDelay:] = basePulse[:-discreteDelay]
    return delayedPulse


def plotWave(wave, fig, axes, title=''):
    axes.set_title(title)
    im = axes.imshow(wave.detach().cpu().numpy(), interpolation='nearest', animated=True, cmap=plt.cm.binary)
    cbar = fig.colorbar(im, ax=axes)
    cbar.ax.set_ylabel('Pressure [Pa]', rotation=270)
    fig.show()
    fig.canvas.draw()
    fig.canvas.flush_events()


def elementOnCircles(num_elements, a, b, grid):
    theta = torch.linspace(0.0, 2 * np.pi, num_elements) # elliptic
    x = a * torch.sin(theta) + grid.nx / 2
    z = -b * torch.cos(theta) + grid.nz / 2
    locations = torch.stack((z, x)).transpose(1, 0)
    return locations

def half_circularMask(num_elements, a, b, grid):
    theta = torch.linspace(0.0, 2 * np.pi, 2*num_elements)  # elliptic
    x = a * np.abs(torch.sin(theta)) + grid.nx / 2
    z = -b * torch.cos(theta) + grid.nz / 2
    locations = torch.stack((z, x)).transpose(1, 0)
    return locations

def parbula_location(num_elements,a, b,c, grid):
    bound = 1
    z = torch.linspace(-bound, bound, num_elements)  # elliptic
    # z = torch.linspace(10, 10, num_elements)  # elliptic
    x =a*z**2+b*z+c
    locations = torch.stack((z, x)).transpose(1, 0)
    return locations

def linear_location(num_elements,a, b, grid):
    # z = torch.linspace(-grid.nx/20, grid.nx/20, num_elements)
    # z = torch.linspace(grid.PML_size+1, grid.nx-1-grid.PML_size, num_elements)
    z = torch.linspace(grid.PML_size+20, grid.nx-20-grid.PML_size, num_elements)

    x = a*z +b
    # x = torch.zeros(16)
    locations = torch.stack((x, z)).transpose(1, 0)
    return locations

def circularMask(a, b, grid):
    x = torch.arange(-grid.nx / 2, grid.nx / 2)
    z = torch.arange(-grid.nz / 2, grid.nz / 2)
    Z, X = torch.meshgrid([z, x])
    mask = 1.0 * ((X**2 / a**2 + Z**2 / b**2) <= 1.0)
    return mask


def gaussianMask(a, b, sigma, grid):
    x = torch.arange(-grid.nx / 2, grid.nx / 2)
    z = torch.arange(-grid.nz / 2, grid.nz / 2)
    Z, X = torch.meshgrid([z, x])
    R = (X**2 / a**2 + Z**2 / b**2).sqrt()
    mask = torch.exp(-R / (2 * sigma)) * (R <= 1.0)
    return mask
