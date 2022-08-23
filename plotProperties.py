import utils

# Initializing the simulator
f0 = 1  # GHz
grid, prob, physicalModel = utils.initializeSimulator(f0)

# Defining the GT velocities
GTProperties = utils.getGTProperties(grid, prob, physicalModel)
num = '1'
propertiesPred = utils.loadProperties(num)

utils.plotProperties(propertiesPred, GTProperties, False, grid, prob, physicalModel)
print('RMSE: c0,               rho0,              D,                 beta')
print('     ', utils.RMSE(GTProperties.velocity.cpu(), propertiesPred.velocity), utils.RMSE(GTProperties.density.cpu(), propertiesPred.density),
      utils.RMSE(GTProperties.damping.cpu(), propertiesPred.damping) * 1e5, utils.RMSE(GTProperties.beta.cpu(), propertiesPred.beta))
