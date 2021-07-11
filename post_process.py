from train import UNet
import torch
import pydensecrf.densecrf as dcrf
import pydensecrf.utils as utils


def morphological_postprocessing(imgs):
    out = []
    for img in imgs:
        kernel = np.ones((3,3), np.uint8)
        img = np.float32(img)
        img = cv2.erode(img, kernel, iterations=1)
        img = cv2.dilate(img, kernel, iterations=1)
        out.append(img)
    return out

def main():
    model = UNet()
    model.load_state_dict(torch.load("model.pth"))
    model.eval()

    # predict on test set
    test_path = 'test'
    test_filenames = (glob(test_path + '/*.png'))
    test_images = load_all_from_path(test_path)
    # batch_size = test_images.shape[0]
    size = test_images.shape[1:3]
    # we also need to resize the test images. This might not be the best ideas depending on their spatial resolution.
    test_images = np.stack([cv2.resize(img, dsize=(384, 384)) for img in test_images], 0)
    test_images = np_to_tensor(np.moveaxis(test_images, -1, 1), device)

    test_pred = []

    for t in test_images.unsqueeze(1):
        y_hat = model(t).detach().cpu().numpy()
        crf_output = np.zeros(y_hat.shape)
        images = t.data.cpu().numpy().astype(np.uint8)
        for i, (image, prob_map) in enumerate(zip(images, y_hat)):
            image = image.transpose(1, 2, 0)
            crf_output[i] = dense_crf(image, prob_map)
        y_hat = crf_output

        test_pred.append(y_hat)


    test_pred = np.concatenate(test_pred, 0)
    test_pred = np.moveaxis(test_pred, 1, -1)  # CHW to HWC
    test_pred = np.stack([cv2.resize(img, dsize=size) for img in test_pred], 0)  # resize to original shape
    
    # morphological postprocessing
    
    kernel = np.ones((3, 3), np.uint8)
    test_pred = np.stack([cv2.erode(img, kernel, iterations=7) for img in test_pred], 0)
    test_pred = np.stack([cv2.dilate(img, kernel, iterations=7) for img in test_pred], 0)
    
    # now compute labels
    test_pred = test_pred.reshape((-1, size[0] // PATCH_SIZE, PATCH_SIZE, size[0] // PATCH_SIZE, PATCH_SIZE))
    test_pred = np.moveaxis(test_pred, 2, 3)
    test_pred = np.round(np.mean(test_pred, (-1, -2)) > CUTOFF)
    # create_submission(test_pred, test_filenames, submission_filename='unet_submission.csv')

    #test_pred = morphological_postprocessing(test_pred)

    create_submission(test_pred, test_filenames, submission_filename='unet_postprocessed_submission.csv')


if __name__ == '__main__':
    main()
