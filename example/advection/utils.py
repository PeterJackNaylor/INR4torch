
import numpy as np
import torch
import matplotlib.pylab as plt

def plot_advection_results(model, hp):
    gt = hp.data_u
    xx = model.test_set.x
    tt = model.test_set.t
    with torch.no_grad():
        predictions = model.test_loop()
    predictions = predictions.float().cpu()
    predictions = predictions.reshape((xx.shape[0], tt.shape[0])).numpy()
    
    # NN.data.nv_targets[0][1] is a normalising constant
    mse = np.linalg.norm(gt - predictions) ** 2 / predictions.shape[0] / model.data.nv_targets[0][1]
    print(f"MSE error: {mse:.6f}")
    
    score = np.linalg.norm(gt - predictions) / np.linalg.norm(gt)
    print(f"Relative L2 error: {score:.6f}")
    
    # Create figure with subplots
    plt.figure(figsize=(12, 12))
    
    # Plot 1: NN prediction
    plt.subplot(3, 3, 1)
    im = plt.imshow(
        predictions,
        extent=[tt.min(), tt.max(), xx.min(), xx.max()],
        aspect="auto",
        cmap="jet",
    )
    plt.colorbar(im)
    plt.title("NN prediction")
    
    # Plot 2: Ground Truth
    plt.subplot(3, 3, 2)
    im = plt.imshow(
        gt, extent=[tt.min(), tt.max(), xx.min(), xx.max()], aspect="auto", cmap="jet"
    )
    plt.colorbar(im)
    plt.title("Ground Truth")
    
    # Plot 3: Absolute error
    plt.subplot(3, 3, 3)
    im = plt.imshow(
        np.abs(gt - predictions),
        extent=[tt.min(), tt.max(), xx.min(), xx.max()],
        aspect="auto",
        cmap="jet",
    )
    plt.colorbar(im)
    plt.title("Absolute error: Gt-Pred")
    
    # Plot 4: Prediction at t=0
    plt.subplot(3, 3, 4)
    plt.plot(predictions[:, 0])
    plt.title("Prediction with t=0")
    
    # Plot 5: Ground truth at t=0
    plt.subplot(3, 3, 5)
    plt.plot(gt[:, 0])
    plt.title("Ground truth with t=0")
    
    # Plot 6: Training error
    plt.subplot(3, 3, 6)
    n = len(model.test_scores)
    f = hp.test_frequency
    plt.plot(list(range(1 * f, (n + 1) * f, f)), model.test_scores)
    plt.title("Training error")
    
    # Plot 7-9: Loss values
    for i, k in enumerate(model.loss_values.keys(), start=7):
        try:
            loss_k = model.loss_values[k]
            plt.subplot(3, 3, i)
            plt.plot(loss_k)
            plt.title(f"Loss {k}")
        except:
            print(f"Couldn't plot {k}")
    
    plt.tight_layout()  # Adjust spacing between subplots
    plt.show()
    plt.close()
