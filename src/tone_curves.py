import numpy as np
import cv2
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

class ToneCurves:
    """Extracts and applies characteristic tone curves from film samples. 
    Will learn the shape the film's contrast curve and store the information as mathematical parameters
    (e.g., for fitting to a sigmoid curve, it would learn the midpoint, steepness, and max value).
    Sigmoid and gamma curves are used for modeling because they capture the naturally S-shaped curve of film contrast 
    """

    def __init__(self):
        self.curve_params = None

    @staticmethod
    def sigmoid_curve(x, midpoint, steepness, max_val):
        '''
        Sigmoid (S-shaped) tone curve function applied to pixel values to model film contract characteristics.
        Input:
            x: input pixel brightness value (0-255)
            midpoint: the input value at which the curve transitions from low to high output (controls 
            where the "S" shape occurs)
            steepness: controls how quickly the curve transitions from low to high output (higher = more contrast)
            max_val: the maximum output value of the curve (controls overall brightness)
        Return: 
            output pixel value after applying the sigmoid curve
        '''
        return max_val / (1 + np.exp(-steepness * (x - midpoint)))

    @staticmethod 
    def gamma_curve(x, gamma):
        '''
        Simple gamma curve function applied to pixel values to model film contrast characteristics.
        Used as fallback if sigmoid fitting fails, due to it's simplicity (single parameter of overall brightness)
        Input:
            x: input pixel brightness value (0-255)
            gamma: controls the shape of the curve; gamma < 1 brightens the image, gamma > 1 darkens the image

        Return:
            output pixel value after applying the gamma curve
        '''
        return 255 * np.power(x / 255.0, gamma)

    def analyze_film_contrast(self, film_samples: list):
        '''
        Analyze contrast characteristics from film samples by extracting luminance histograms and fitting a 
        curve to the cumulative histogram to learn the film's tone curve parameters.
        Looking for: how does the film react to lighting conditions (shadows, highlights)?

        Input:
            film_samples: list of film images in RGB format

        Output: 
            None (stores learned parameters in variable self.curve_params)
        '''
        # Extract luminance histograms 
        luminance_samples = []
        for img in film_samples:
            # convert to grayscale for luminance analysis by averageing RGB channels (R+G+B)/3
            # Will give single brightness value per pixel (0-255)
            gray = np.mean(img, axis=2) # axis=2 means average across color channels (img size = `height (axis = 0) x width (axis = 1) x 3 channels (RGB) (axis = 2)`)
            luminance_samples.append(gray)

        hist = np.zeros(256) # initialize histogram with 256 bins for pixel values 0-255
        hist_sum = np.zeros(256) # initialize array to sum histograms across all film samples
        for lum in luminance_samples:
            # compute histogram for each image luminance and sum to get overall luminance for film stock
            hist, _ = np.histogram(lum.flatten(), bins = 256, range=(0, 255))
            hist_sum += hist # sum histograms across all film samples to get overall luminance distribution for the film stock 

        avg_hist = hist_sum / len(luminance_samples) # average histogram across all film samples to get distribution of luminance values for the film stock 

        # using cumulative sum naturally gives an S-shaped curve, which is perfect for fitting a sigmoid function
        # this is used to learn the parameters of the film's tone curve, which will mimic the contrast characteristics of the film stock
        cumsum = np.cumsum(avg_hist) # estimate curve parameters from histogram shape by fitting to cumulative hist.
        cumsum_normalized = cumsum / cumsum[-1] * 255 # normalize to 0-255 range 

        x_data = np.arange(256) # input pixel brightness values 0-255 
        try:
            # fit sigmoid curve to cumulative hist. to learn film's contrast characteristics 
            # maxfev set to 1k iterations; will resort to gamma curve if sigmoid fit fails after 1k iterations
            # curve_fit returns optimal parameters for the sigmoid curve (midpoint, steepness, max_val) 
            # curve_fit also returns covariance matrix, which can be ignored in this case
            params, _ = curve_fit(self.sigmoid_curve, x_data, cumsum_normalized, p0 = [128, 0.05, 255], maxfev=10000) 
            self.curve_params = {
                'type': 'sigmoid', 
                'midpoint': params[0],
                'steepness': params[1],     
                'max_val': params[2]
            }
        except:
            # fallback incase sigmoid fitting fails (due to lack of contrast or other issues with histogram shape)
            print("Sigmoid fit failed, using gamma curve")
            self.curve_params = {
                'type': 'gamma',
                'gamma': 1.2 # default gamma value for fallback 
            }

        print(f"Extracted curve parameters: {self.curve_params}")


    def analyze_curve(self, image: np.ndarray, strength: float = 1.0) -> np.ndarray:
        '''
        Apply learned tone curve to digital image 

        Input: 
            image: digital image in RGB format 
            strength: how strongly to apply the curve (0-1); 1 = full contrast, 0 = no change

        Return: 
            image with tone curve applied to mimic film contrast characteristics
        '''
        if self.curve_params is None:
            raise ValueError("Must call analyze_film_contrast() first to learn curve parameters")
        
        lut = np.arange(256, dtype=np.float32) # create lookup table for pixel values 0-255 (using array for vectorization purposes)

        # pass all 256 input values through curve function
        # lut now contains new transformed values for contrast
        if self.curve_params['type'] == 'sigmoid':
            lut = self.sigmoid_curve(lut, self.curve_params['midpoint'], self.curve_params['steepness'], self.curve_params['max_val'])
        else:
            lut = self.gamma_curve(lut, self.curve_params['gamma'])

        # blend with linear (original) curve based on strength parameter to allow for adjustable contrast
        original_curve = np.arange(256, dtype=np.float32)
        lut = original_curve * (1 - strength) + lut * strength # linearly blend original and film curve based on strength parameter (0 = original, 1 = full film curve)
        lut = np.clip(lut, 0, 255).astype(np.uint8) # ensure values are in valid pixel range and convert to uint8 for cv2 LUT function

        # apply tone curve to image using openCV LUT; maps each pixel value in the image to its corresponding value in the LUT to create the output image with adjusted contrast
        result = cv2.LUT(image, lut) 
        return result

    def visualization(self, film_samples: list):
        ''' 
        Plot learned tone curve against the original cumulative histogram of film samples.
        Will show: 
            Red dashed line = linear curve (original input pixel values)
            Blue solid line = learned film curve (after fitting to histogram)

        Input:
            film_samples: list of film images in RGB format

        Return:
            None (displays plot)
        '''
        if self.curve_params is None:
            return("No curve to visualize; must call analyze_film_contrast() to learn curve parameters")
        
        x = np.arange(256)
        # 
        if self.curve_params['type'] == 'sigmoid':
            y = self.sigmoid_curve(x, self.curve_params['midpoint'], self.curve_params['steepness'], self.curve_params['max_val'])
        else:
            y = self.gamma_curve(x, self.curve_params['gamma'])
            
        plt.figure(figsize=(8, 6))
        plt.plot(x, x, 'r--', label='Linear (Original)', alpha=0.5)
        plt.plot(x, y, 'b-', label='Film Curve', linewidth=2)
        plt.xlabel('Input Value')
        plt.ylabel('Output Value')
        plt.title('Tone Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    