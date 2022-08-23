import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import timeit

import utils


class SoundWave2DModel(torch.nn.Module):
    def __init__(self, properties, grid, prob, physicalModel):
        super(SoundWave2DModel, self).__init__()
        self.grid = grid
        self.prob = prob
        self.physicalModel = physicalModel

        self.velocity = torch.nn.Parameter(properties.velocity.detach().clone().to(grid.dtype).to(grid.device))
        self.density = torch.nn.Parameter(properties.density.detach().clone().to(grid.dtype).to(grid.device))
        self.damping = torch.nn.Parameter(properties.damping.detach().clone().to(grid.dtype).to(grid.device))
        self.beta = torch.nn.Parameter(properties.beta.detach().clone().to(grid.dtype).to(grid.device))

        self.loc = torch.nn.Parameter(properties.loc.detach().clone().to(grid.dtype).to(grid.device)).type(torch.long)
        if not grid.reconstructVelocity: self.velocity.requires_grad = False
        if not grid.reconstructDensity: self.density.requires_grad = False
        if not grid.reconstructDamping: self.damping.requires_grad = False
        if not grid.reconstructBeta: self.beta.requires_grad = False

        # Each iteration is obtained by a convolution with the Laplacian filter
        laplacian_kernel = torch.tensor([[0, 0, -1. / 12, 0, 0],
                                         [0, 0, 4. / 3, 0, 0],
                                         [-1. / 12, 4. / 3, -2 * 5. / 2, 4. / 3, -1. / 12],
                                         [0, 0, 4. / 3, 0, 0],
                                         [0, 0, -1. / 12, 0, 0]], dtype=grid.dtype, device=grid.device)
        self.cnn_layer_laplacian = nn.Conv2d(1, 1, kernel_size=(grid.nop, grid.nop),
                                             padding=(int((grid.nop - 1) / 2), int((grid.nop - 1) / 2)))
        self.cnn_layer_laplacian.weight = torch.nn.Parameter(
            laplacian_kernel[None, None, :, :].detach().to(grid.dtype).to(grid.device))
        self.cnn_layer_laplacian.weight.requires_grad = False
        self.cnn_layer_laplacian.bias = None
        self.laplacian = lambda x: 1 / self.grid.dz ** 2 * self.cnn_layer_laplacian(x)

        grad_z_kernel = 0.5 * torch.tensor([[0, -1, 0],
                                      [0, 0, 0],
                                      [0, 1, 0]], dtype=grid.dtype, device=grid.device)
        self.cnn_layer_grad_z = nn.Conv2d(1, 1, kernel_size=(grid.nop, grid.nop), padding=(1, 1))
        self.cnn_layer_grad_z.weight = torch.nn.Parameter(
            grad_z_kernel[None, None, :, :].detach().to(grid.dtype).to(grid.device))
        self.cnn_layer_grad_z.weight.requires_grad = False
        self.cnn_layer_grad_z.bias = None
        self.grad_z = lambda x: 1 / self.grid.dz * self.cnn_layer_grad_z(x)

        grad_x_kernel = 0.5 * torch.tensor([[0, 0, 0],
                                      [-1, 0, 1],
                                      [0, 0, 0]], dtype=grid.dtype, device=grid.device)
        self.cnn_layer_grad_x = nn.Conv2d(1, 1, kernel_size=(grid.nop, grid.nop), padding=(1, 1))
        self.cnn_layer_grad_x.weight = torch.nn.Parameter(
            grad_x_kernel[None, None, :, :].detach().to(grid.dtype).to(grid.device))
        self.cnn_layer_grad_x.weight.requires_grad = False
        self.cnn_layer_grad_x.bias = None
        self.grad_x = lambda x: 1 / self.grid.dz * self.cnn_layer_grad_x(x)

        diag1_kernel = 0.5 * torch.tensor([[0, 0, 0],
                                           [0, -1, 0],
                                           [0,  0, 1]], dtype=grid.dtype, device=grid.device)
        self.cnn_layer_diag1 = nn.Conv2d(1, 1, kernel_size=(grid.nop, grid.nop), padding=(0, 0))
        self.cnn_layer_diag1.weight = torch.nn.Parameter(diag1_kernel[None, None, :, :].detach().to(grid.dtype).to(grid.device))
        self.cnn_layer_diag1.weight.requires_grad = False
        self.cnn_layer_diag1.bias = None
        self.diag1 = lambda x: 1 / self.grid.dz * self.cnn_layer_diag1(x)

        diag2_kernel = 0.5 * torch.tensor([[0, 0, 0],
                                           [0, -1, 0],
                                           [1,  0, 0]], dtype=grid.dtype, device=grid.device)
        self.cnn_layer_diag2 = nn.Conv2d(1, 1, kernel_size=(3, 3), padding=(0, 0))
        self.cnn_layer_diag2.weight = torch.nn.Parameter(diag2_kernel[None, None, :, :].detach().to(grid.dtype).to(grid.device))
        self.cnn_layer_diag2.weight.requires_grad = False
        self.cnn_layer_diag2.bias = None
        self.diag2 = lambda x: 1 / self.grid.dz * self.cnn_layer_diag2(x)

        # sobelx_kernel = 0.5 * torch.tensor([[1, 0, -1],
        #                                    [2, 0, -2],
        #                                    [1,  0, -1]], dtype=grid.dtype, device=grid.device)
        sobelx_kernel = 0.5 * torch.tensor([[0, 0, 0],
                                            [0, -1, 1],
                                            [0,  0, 0]], dtype=grid.dtype, device=grid.device)
        self.cnn_layer_sobelx = nn.Conv2d(1, 1, kernel_size=(3, 3), padding=(0, 0))
        self.cnn_layer_sobelx.weight = torch.nn.Parameter(sobelx_kernel[None, None, :, :].detach().to(grid.dtype).to(grid.device))
        self.cnn_layer_sobelx.weight.requires_grad = False
        self.cnn_layer_sobelx.bias = None
        self.sobelx = lambda x: 1 / self.grid.dz * self.cnn_layer_sobelx(x)

        # sobelz_kernel = 0.5 * torch.tensor([[1, 2, 1],
        #                                    [0, 0, 0],
        #                                    [-1,  -2, -1]], dtype=grid.dtype, device=grid.device)
        sobelz_kernel = 0.5 * torch.tensor([[0, 0, 0],
                                            [0, -1, 0],
                                            [0,  1, 0]], dtype=grid.dtype, device=grid.device)
        self.cnn_layer_sobelz = nn.Conv2d(1, 1, kernel_size=(3, 3), padding=(0, 0))
        self.cnn_layer_sobelz.weight = torch.nn.Parameter(sobelz_kernel[None, None, :, :].detach().to(grid.dtype).to(grid.device))
        self.cnn_layer_sobelz.weight.requires_grad = False
        self.cnn_layer_sobelz.bias = None
        self.sobelz = lambda x: 1 / self.grid.dz * self.cnn_layer_sobelz(x)

        deconvolution_kernel = torch.from_numpy(prob.short_basePulse).to(dtype=grid.dtype).to(device=grid.device)[:, None]
        deconvolution_kernel = deconvolution_kernel / deconvolution_kernel.abs().sum()
        self.cnn_layer_deconvolution = nn.Conv2d(1, 1, kernel_size=(deconvolution_kernel.shape[0], deconvolution_kernel.shape[1]), padding='same')
        self.cnn_layer_deconvolution.weight = torch.nn.Parameter(deconvolution_kernel[None, None, :, :].detach().to(grid.dtype).to(grid.device))
        self.cnn_layer_deconvolution.weight.requires_grad = False
        self.cnn_layer_deconvolution.bias = None
        self.deconvolution = lambda x: self.cnn_layer_deconvolution(x[None, None, :, :])[0][0]

        self.grad_dot_grad = lambda x, y: self.grad_x(1 / (x[None, None, :, :])) * self.grad_x(y) + self.grad_z(
            1/(x[None, None, :, :])) * self.grad_z(y)

        self.sobel  = lambda x: (self.sobelx(x[None, None, :, :]).abs() + self.sobelz(x[None, None, :, :]).abs()) #   self.sobel  = lambda x: (self.sobelz(x[None, None, :, :]).abs()) #
        self.robert = lambda x: (self.diag1(x[None, None, :, :]).abs() + self.diag2(x[None, None, :, :]).abs())

    def forward(self, pulse, NLAmodel, plot=False):
        # intensity = torch.zeros([1, 1, self.grid.nz, self.grid.nx], dtype=self.grid.dtype, device=self.grid.device)
        p = torch.zeros([1, 1, self.grid.nz, self.grid.nx], requires_grad=True, dtype=self.grid.dtype, device=self.grid.device)
        pm1 = torch.zeros([1, 1, self.grid.nz, self.grid.nx], requires_grad=True, dtype=self.grid.dtype, device=self.grid.device)
        pm2 = torch.zeros([1, 1, self.grid.nz, self.grid.nx], requires_grad=True, dtype=self.grid.dtype, device=self.grid.device)
        trans = []
        ir = np.arange(self.prob.numChannels)

        K = lambda c, rho: c.pow(2) * rho
        NLA =  lambda beta, c, rho, pm1: (1 + 2 * beta / K(c, rho) * pm1)
        NLA2 = lambda beta, c, rho: (2 * beta / K(c, rho))

        if not self.grid.inversion:
            ps = torch.zeros([self.grid.nt, 1, 1, self.grid.nz, self.grid.nx], dtype=self.grid.dtype, device=self.grid.device)

        for t in range(self.grid.nt):
            if NLAmodel:
                dt = self.grid.dt
                a = NLA(self.beta, self.velocity, self.density, pm1) + 2 * self.damping
                b = (2 * NLA(self.beta, self.velocity, self.density, pm1) - self.damping.pow(2))
                c = (2 * self.damping - NLA(self.beta, self.velocity, self.density, pm1))
                d = NLA2(self.beta, self.velocity, self.density)

                p = (b * pm1 + c * pm2 + d * (pm1 - pm2).pow(2) +
                     dt.pow(2) * K(self.velocity, self.density) * (self.density * self.laplacian(p) + self.grad_dot_grad(self.density, p))
                     + pulse[t]) / a
                # intensity += p.pow(2)
            else: # Linear model
                dt = self.grid.dt
                a = 1 + 2 * self.damping
                b = 2 - self.damping.pow(2)
                c = 2 * self.damping - 1

                p = (b * pm1 + c * pm2 +
                     dt.pow(2) * K(self.velocity, self.density) * (self.density * self.laplacian(p) + self.grad_dot_grad(self.density, p))
                     + dt.pow(2) * pulse[t]) / a
                # intensity += p.pow(2)
            pm2, pm1 = pm1, p

            # Save signal on transducers
            if t % self.grid.oversampling_time == 0:
                #trans.append(p[0][0][self.prob.elementsLocations[ir, 0], self.prob.elementsLocations[ir, 1]]) #OLD
                trans.append(p.clone()[0][0][self.loc[ir, 0], self.loc[ir, 1]])

            if not self.grid.inversion:
                ps[t] = p #self.deconvolution(p[0][0])

        if self.grid.inversion:
            trans_stack = torch.stack(trans)
            return trans_stack #self.deconvolution( trans_stack )
        else:
            return ps


def SoundWave2D(properties, pulse, grid, prob, physicalModel, plot=False):
    ir = np.arange(prob.numChannels)
    trans = torch.zeros([prob.numChannels, grid.nt], dtype=grid.dtype, device=grid.device)  # The signal on the transducers

    # Initialize animated plot
    if plot:
        prob.pulses = pulse
        fig, axes, im = prepareFigWave2D(properties, grid, prob, physicalModel)

    model = SoundWave2DModel(properties, grid, prob, physicalModel)
    if grid.device is torch.device("cuda"):
        model = model.cuda()

    if plot: ## plot the channel data
        ps = model.forward(pulse, NLAmodel=True)
        for it in range(grid.nt):
            #trans[ir, it] = ps[it][0][0][prob.elementsLocations[ir, 0], prob.elementsLocations[ir, 1]] #OLD
            trans[ir, it] = ps[it][0][0][model.loc[ir, 0], model.loc[ir, 1]]
            plotWave2D(ps[it], properties, trans, it, plot, ir, fig, axes, im, grid, prob, physicalModel)
        return ps
    return model.forward(pulse, NLAmodel=True)


def prepareFigWave2D(properties, grid, prob, physicalModel):
    fig = plt.figure(figsize=(10, 24))
    axes = []
    nrow = 1
    ncol = 6
    axes.append(plt.subplot2grid((nrow, ncol), (0, 0)))
    axes.append(plt.subplot2grid((nrow, ncol), (0, 1)))
    axes.append(plt.subplot2grid((nrow, ncol), (0, 2)))
    axes.append(plt.subplot2grid((nrow, ncol), (0, 3)))
    axes.append(plt.subplot2grid((nrow, ncol), (0, 4)))
    axes.append(plt.subplot2grid((nrow, ncol), (0, 5)))

    toPlot = torch.zeros([grid.nz, grid.nx])

    extent = [-100 * grid.Lx / 2, 100 * grid.Lx / 2, -100 * grid.Lz / 2, 100 * grid.Lz / 2]
    for ii in range(5):
        axes[ii].set_xlabel('cm')
        axes[ii].set_ylabel('cm')
    vmax_pressure = 0.01 * (prob.pulses.max()).to(torch.float64).item()# / grid.dt.detach()**2# / max([np.abs(prob.pulses.min().cpu()), np.abs(prob.pulses.max().cpu())])
    im = []
    im.append(axes[0].imshow(toPlot, interpolation='nearest', animated=True, vmin=-vmax_pressure, vmax=vmax_pressure,
                             cmap=plt.cm.RdBu, extent=extent))
    cbar = fig.colorbar(im[0], ax=axes[0])
    cbar.ax.set_ylabel('Pressure [Pa]', rotation=270)
    im.append(axes[1].imshow(toPlot, interpolation='nearest', animated=True, vmin=1e6 * physicalModel.vmin_velocity,
                             vmax=1e6 * physicalModel.vmax_velocity, cmap=plt.cm.RdBu, extent=extent))
    cbar = fig.colorbar(im[1], ax=axes[1])
    cbar.ax.set_ylabel('[m / s]', rotation=270)
    im.append(axes[2].imshow(toPlot, interpolation='nearest', animated=True, vmin=physicalModel.vmin_density,
                             vmax=physicalModel.vmax_density, cmap=plt.cm.RdBu, extent=extent))
    cbar = fig.colorbar(im[2], ax=axes[2])
    cbar.ax.set_ylabel('[1000 * kg / m^3]', rotation=270)
    im.append(axes[3].imshow(toPlot, interpolation='nearest', animated=True, vmin=physicalModel.vmin_damping,
                             vmax=physicalModel.vmax_damping, cmap=plt.cm.RdBu, extent=extent))
    cbar = fig.colorbar(im[3], ax=axes[3])
    cbar.ax.set_ylabel('[1 / s]', rotation=270)
    im.append(axes[4].imshow(toPlot, interpolation='nearest', animated=True, vmin=physicalModel.vmin_beta,
                             vmax=physicalModel.vmax_beta, cmap=plt.cm.RdBu, extent=extent))
    cbar = fig.colorbar(im[4], ax=axes[4])
    cbar.ax.set_ylabel('', rotation=270)
    im.append(axes[5].imshow(toPlot, interpolation='nearest', animated=True, vmin=physicalModel.vmin_location,
                             vmax=physicalModel.vmax_location, cmap='gray', extent=extent))
    cbar = fig.colorbar(im[5], ax=axes[5])
    cbar.ax.set_ylabel('', rotation=270)

    toPlot = properties.velocity.detach().cpu()
    im[1].set_data(1e6 * toPlot)
    toPlot = properties.density.detach().cpu()
    im[2].set_data(toPlot)
    toPlot = properties.damping.detach().cpu()
    im[3].set_data(toPlot)
    toPlot = properties.beta.detach().cpu()
    im[4].set_data(toPlot)
    toPlot = properties.loc.detach().cpu()
    im[5].set_data(toPlot)

    model =  SoundWave2DModel(properties, grid, prob, physicalModel) # I added# I added

    for s in range(prob.numChannels):
        #axes[0].text(prob.elementsPhysicalLocations[s, 1] * 100, prob.elementsPhysicalLocations[s, 0] * 100, 'o') #OLD
        axes[0].text(model.loc[s, 1], model.loc[s, 0], 'o') ###???

    axes[0].set_title('Acoustic Wave')
    axes[1].set_title('Velocity')
    axes[2].set_title('Density')
    axes[3].set_title('Damping')
    axes[4].set_title('Beta')
    axes[5].set_title('Location')


    return fig, axes, im


def plotWave2D(p, properties, trans, it, plot, ir, fig, axes, im, grid, prob, physicalModel):
    isnap = 10  # snapshot frequency

    if it % isnap == 0 and plot:

        toPlot = p[-1].detach().cpu()
        toPlot = toPlot[0, :, :]
        im[0].set_data(toPlot)

        fig.set_figheight(10)
        fig.set_figwidth(20)
        plt.gcf().canvas.draw()
        plt.ion()
        plt.show()
        plt.gcf().canvas.flush_events()
        if grid.inversion:
            fig.savefig('optimizationProcess_' + str(it) + '.png', dpi=fig.dpi)


def preparePlotOptimization(GTProperties, grid, prob, physicalModel):
    fig, axes, im = utils.plotProperties(GTProperties, GTProperties, True, grid, prob, physicalModel)
    axes[1, 5].set_title('Loss [dB]')
    return fig, axes, im


def plotOptimizationProcess(model, loss, losses, t, fig, axes, im,  grid, prob, physicalModel):
    velocityFiltered = model.velocity.detach().cpu().numpy()
    densityFiltered = model.density.detach().cpu().numpy()
    dampingFiltered = model.damping.detach().cpu().numpy()
    betaFiltered = model.beta.detach().cpu().numpy()
    locationFiltered = model.loc.detach().cpu().numpy() ## i added


    fig.suptitle('Backpropogating to the velocities, iter = ' + str(t) + ', time = ' + str(grid.time) + ', loss = ' + str(loss.item()))
    im[0].set_data(velocityFiltered)
    im[1].set_data(locationFiltered[1],locationFiltered[0])
    im[2].set_data(densityFiltered)

    im[3].set_data(dampingFiltered)
    im[4].set_data(betaFiltered)
    # im[4].set_data(locationFiltered[0],locationFiltered[1])

    # im[1].set_data(densityFiltered)
    # im[2].set_data(dampingFiltered)
    # im[3].set_data(betaFiltered)
    # im[4].set_data(locationFiltered[0],locationFiltered[1])

    axes[1, 5].plot(np.log(losses), color='darkslategrey')
    fig.show()
    fig.canvas.draw()
    fig.canvas.flush_events()
    fig.savefig('OptimizationProcess/optimizationProcess_' + str(t) + '.png', dpi=fig.dpi)


def allTransmission(initialProperties, GTProperties, grid, prob, physicalModel):
    grid.noise = [n.to(device=grid.device) for n in grid.noise]
    grid.mask = grid.mask.to(device=grid.device)

    channelData = []
    for lateral in range(prob.numLaterals):
        pulse = pulseAllTransmision(grid, prob, lateral,GTProperties,physicalModel)
        channelData.append( SoundWave2D(GTProperties, pulse, grid, prob, physicalModel, plot=not grid.inversion).detach())# + grid.noise[lateral])
    channelData_stack = torch.stack(channelData)

    plotProcess = True
    propertiesPred = utils.prop2dict( InverseUSSolver_stack(initialProperties, channelData_stack, GTProperties, plotProcess, grid, prob, physicalModel) )

    mask_base = grid.mask_base.to('cuda:0')
    propertiesPred['velocity'] = propertiesPred['velocity'] * mask_base + physicalModel.c0 * (1 - mask_base)
    propertiesPred['density'] = propertiesPred['density'] * mask_base + 1.0 * (1 - mask_base)
    propertiesPred['damping'] = propertiesPred['damping'] * mask_base
    propertiesPred['beta'] = propertiesPred['beta'] * mask_base
    propertiesPred['location'] = propertiesPred['location'] * mask_base


    propertiesPred['velocity'].clamp_(physicalModel.vmin_velocity, physicalModel.vmax_velocity)
    propertiesPred['density'].clamp_(physicalModel.vmin_density, physicalModel.vmax_density)
    propertiesPred['damping'].clamp_(physicalModel.vmin_damping, physicalModel.vmax_damping)
    propertiesPred['beta'].clamp_(physicalModel.vmin_beta, physicalModel.vmax_beta)
    # propertiesPred['location'].clamp_(physicalModel.vmin_location, physicalModel.vmax_location)

    return propertiesPred


def InverseUSSolver_stack(initialProperties, trans_stack, GTvelocities, plotProcess, grid, prob, physicalModel):

    model, optimizer, scheduler = getModel(initialProperties, grid, prob, physicalModel, 1e0)

    mseLoss = nn.MSELoss()
    l1Loss = nn.L1Loss()

    if plotProcess:
        fig, axes, im = preparePlotOptimization(GTvelocities, grid, prob, physicalModel)

    losses = []
    epsVelocity = 0.0001
    epsDensity = 0.0001
    epsDamping = 0.0
    epsBeta = 0.0
    epsLoc = 0.0

    if grid.rep == 0:
        numItrations = 150
        alpha = 8e-0
        beta = 6e-2
        loss_fn = lambda x, y: 1e3 * l1Loss(x, y)
    else:
        numItrations = 50
        alpha = 4e-10
        beta = 6e-4
        loss_fn = lambda x, y: 1e4 * mseLoss(x, y)


    startTime = timeit.default_timer()
    for t in range(numItrations):
        transPred = []
        for lateral in range(prob.numLaterals):
            pulse = pulseAllTransmision(grid, prob, lateral,initialProperties,physicalModel)
            transPred.append( model(pulse, grid.NLA_model) )
        transPred_stack = torch.stack(transPred)

        reg = lambda x: model.robert(x).mean() + model.sobel(x).mean()
        loss = loss_fn(transPred_stack, trans_stack) + alpha * reg(model.velocity) + beta * reg(model.density)
        losses.append(loss.item())
        print('iteration', str(t), ':', loss_fn(transPred_stack, trans_stack).item(), alpha * reg(model.velocity).item(), beta * reg(model.density).item())

        optimizer.zero_grad()
        loss.backward()

        if plotProcess:
            grid.time = timeit.default_timer() - startTime
            plotOptimizationProcess(model, loss, losses, t + 1, fig, axes, im, grid, prob, physicalModel)

        if grid.reconstructVelocity:
            # model.velocity.grad = model.deconvolution(model.velocity.grad)
            grad_velocity = model.velocity.grad
            epsG = (epsVelocity * grad_velocity).abs().max()
            model.velocity.grad = grad_velocity * (grad_velocity.abs() > epsG) * grid.mask
        # else:
        #     model.velocity.grad = 0 * model.velocity.grad

        if grid.reconstructDensity:
            # model.density.grad = model.deconvolution(model.density.grad)
            grad_density = model.density.grad
            epsG = (epsDensity * grad_density).abs().max()
            model.density.grad = grad_density * (grad_density.abs() > epsG) * grid.mask
        # else:
        #     model.density.grad = 0 * model.density.grad

        if grid.reconstructDamping:
            grad_damping = model.damping.grad
            epsG = (epsDamping * grad_damping).abs().max()
            model.damping.grad = grad_damping * (grad_damping.abs() > epsG) * grid.mask
        # else:
        #     model.damping.grad = 0 * model.damping.grad

        if grid.reconstructBeta:
            grad_beta = model.beta.grad
            epsG = (epsBeta * grad_beta).abs().max()
            model.beta.grad = grad_beta * (grad_beta.abs() > epsG) * grid.mask
        # else:
        #     model.beta.grad = 0 * model.beta.grad

        # if grid.reconstructLoc:
        #     grad_loc = model.loc.grad
        #     epsG = (epsLoc * grad_loc).abs().max()
        #     model.beta.grad = grad_loc * (grad_loc.abs() > epsG) * grid.mask
        # # else:
        #     model.loc.grad = 0 * model.loc.grad

    # Updating velocities
        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            model.velocity.clamp_(physicalModel.vmin_velocity, physicalModel.vmax_velocity)
            model.density.clamp_(physicalModel.vmin_density, physicalModel.vmax_density)
            model.damping.clamp_(physicalModel.vmin_damping, physicalModel.vmax_damping)
            model.beta.clamp_(physicalModel.vmin_beta, physicalModel.vmax_beta)
            # model.loc.clamp_(physicalModel.vmin_location, physicalModel.vmax_location) #I added


        if t == numItrations - 1:
            utils.saveProperties('1', model, dir='Results/')

    return model


def getModel(properties, grid, prob, physicalModel, lr):
    newProperties = utils.Properties()
    newProperties.velocity = properties.velocity
    newProperties.density = properties.density
    newProperties.damping = properties.damping
    newProperties.beta = properties.beta
    newProperties.loc = properties.loc # I added

    model = SoundWave2DModel(properties, grid, prob, physicalModel)
    cudaDevice = grid.device
    grid.device = cudaDevice

    model = model.cuda(device=grid.device)
    optimizer = torch.optim.Adam([
        {'params': model.velocity, 'lr': grid.factor * 1e-6},
        {'params': model.density, 'lr': grid.factor * 8e-3},
        {'params': model.damping, 'lr': grid.factor * 3e-9},
        {'params': model.beta, 'lr': grid.factor * 3e-2},
        {'params': model.loc, 'lr': grid.factor * 3e-2}], lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.90)
    return model, optimizer, scheduler


def pulseAllTransmision(grid, prob, lateral,properties,physicalModel):
    elementsApodizations = torch.zeros([1, prob.numChannels]).to(torch.int)
    elementsApodizations[0, lateral] = 1
    prob.elementsApodizations = elementsApodizations

    model = SoundWave2DModel(properties, grid, prob, physicalModel) # I added
    # ir = np.arange(prob.numChannels)
    pulses = torch.zeros([grid.nx, grid.nz, grid.nt], dtype=grid.dtype, device=grid.device)
    for channel in range(prob.numChannels):
        if prob.elementsApodizations[0, channel]:
            pulses[model.loc[channel, 0], model.loc[channel, 1]] = prob.base_pulse.clone().to(grid.dtype).to(grid.device)  ###### was changed########
            #pulses[prob.elementsLocations[channel, 0], prob.elementsLocations[channel, 1]] = prob.base_pulse.clone().to(grid.dtype).to(grid.device) #OLD
    pulses *= 1.0 / pulses.abs().max()
    pulses = pulses.permute(2, 0, 1)
    return pulses


def saveChannelData(GTProperties, grid, prob, plot, physicalModel):
    grid.noise = [n.to(device=grid.device) for n in grid.noise]
    grid.mask = grid.mask.to(device=grid.device)

    channelData = []
    for lateral in range(prob.numLaterals):
        pulse = pulseAllTransmision(grid, prob, lateral,GTProperties,physicalModel)
        channelData.append( SoundWave2D(GTProperties, pulse, grid, prob, physicalModel, plot=plot).detach())# + grid.noise[lateral])
    channelData_stack = torch.stack(channelData)
    np.save('Resutls/Channel_data.npy', channelData_stack.detach().cpu().numpy())
    return channelData_stack