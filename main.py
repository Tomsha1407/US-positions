import numpy as np
import torch
import timeit

import SoundWave2D
import utils


def main():
    inversion = True# If true this will perform the inversion, otherwise it will visualize the wave propagation
    f0 = 7.6  # MHz

    # Initializing the simulator
    grid, prob, physicalModel = utils.initializeSimulator(f0)
    grid.inversion = inversion
    grid.noise = utils.getNoise(grid, prob)

    a_initial = 32
    b_initial = 30
    grid.mask = utils.circularMask(a_initial, b_initial, grid).to(grid.device).to(grid.dtype)
    grid.mask_base = grid.mask.clone().to(grid.device).to(grid.dtype)

    # Defining the GT properties and initial properties
    GTProperties = utils.getGTProperties(grid, prob, physicalModel)
    initialProperties = utils.initializeProperties(0, 0, GTProperties, grid, prob, physicalModel, DASInitialization=False)

    if inversion:
        # At each iteration, the algorithm will optimize over n_l lateral scan.
        # The variable repetition, define how many times this is done.
        repetitions = 1
        propertiesPred = [initialProperties]  # A list containing the predictions
        startTime = timeit.default_timer()
        for ind_r in range(repetitions):
            grid.rep = ind_r
            propertiesPred.append(utils.dict2prop(SoundWave2D.allTransmission(propertiesPred[-1], GTProperties, grid, prob, physicalModel)))
            utils.plotProperties(propertiesPred[-1], GTProperties, False, grid, prob, physicalModel)
            utils.saveProperties('1', propertiesPred[-1])

        stopTime = timeit.default_timer()
        print('Inverse US Solver Time = ', stopTime - startTime)

        print('RMSE: c0,               rho0,              D,                 beta')
        print('     ', utils.RMSE(GTProperties.velocity.cpu(), propertiesPred[-1].velocity.cpu()), utils.RMSE(GTProperties.density.cpu(), propertiesPred[-1].density.cpu()),
              utils.RMSE(GTProperties.damping.cpu(), propertiesPred[-1].damping.cpu()) * 1e5, utils.RMSE(GTProperties.beta.cpu(), propertiesPred[-1].beta.cpu()))
    else:
        plot = True# If true, visualize the wave propagation
        SoundWave2D.saveChannelData(GTProperties, grid, prob, plot, physicalModel)


if __name__ == "__main__":
    main()
