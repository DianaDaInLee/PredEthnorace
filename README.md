Step-by-Step Guidelines to Implement Hybrid Ethnoracial Prediction
================
Diana Da In Lee (<dl2860@columbia.edu>)
Last Updated: 06 August, 2023

This guideline walks through how to implement the proposed hybrid
approach to ethnoracial prediction in Lee, Diana, and Yamil Velez.
“[Measuring descriptive representation at scale: Methods for predicting
the race and ethnicity of public officials.](https://osf.io/tpsv6/)”
(2023+).

## 1. Image-Based Prediction

We first want to collect images associated with each name and make
predictions based on the images. We provide two python functions
`img_search` and `img_pred` to do these steps. For users who don’t have
python installed in their computer, we also provide a Google Colab
demonstration here:
[PredEthnorace_Demo.ipynb](https://github.com/DianaDaInLee/PredEthnorace/blob/main/PredEthnorace_Demo.ipynb).

### 1.1. Collect Images

Collect profile images of the individuals in the dataset, using their
first and last names as a keyword. Users may use any public engines with
few notable ones include:

- Microsoft Bing:
  <https://www.microsoft.com/en-us/bing/apis/bing-image-search-api>
- Google Vision API: <https://cloud.google.com/vision/>
- DuckDuckGo: <https://www.duckduckgo.com>.

Private search APIs also exist that allows to search images in various
engines. For example, [SerpAPI](https://serpapi.com/images-results) is a
real time API to access search results from Bing, DuckDuckGo, Google,
and more. Open-source packages are also available (e.g., see
[duckduckgo-search](https://pypi.org/project/duckduckgo-search/)).

We provide a python function `img_check` that allows for users to check
for the quality of images collected. It checks for the total number of
faces detected as well as any duplicated faces (i.e., detected to be of
a same person). It is basically a wrapper for `deepface` package created
by [Serengil](https://github.com/serengil/deepface). It takes following
arguments:

- `imgfolder` (string): Name of the folder where images are saved.
- `detector` (string): Detector backend for facial recognition. Choose
  from “opencv” (default), “retinaface”, “mtcnn”, “ssd”, “dlib”,
  “mediapipe” or “yolov8”. See <https://github.com/serengil/deepface>
  for differences across different detectors.
- `distance` (string): Distance metric to measure similarity between two
  images. Choose from “cosine”, “euclidean”, “euclidean_l2” (default).
- `model` (string): Face regonition model to use. Choose from
  “VGG-Face”, “Facenet”, “Facenet512”, “OpenFace” (default), “DeepFace”,
  “DeepID”, “ArcFace”, “Dlib”, “SFace”. See
  <https://github.com/serengil/deepface> for differences across
  different models.

Using the demo images,

``` python
a, b = img_check(imgfolder = 'demo')
```

    ##              image n_faces
    ## 1 demo/image2.jpeg       1
    ## 2 demo/image1.jpeg      47
    ## 3 demo/image4.jpeg       1
    ## 4 demo/image3.jpeg       1
    ## 5  demo/image5.jpg       1

We see that `image1.jpeg` contains more than 40 individuals. You might
want to remove image files like this.

    ##              image1           image2  dupe  distance    model detector
    ## 1  demo/image2.jpeg demo/image1.jpeg False 0.5536687 OpenFace   opencv
    ## 2  demo/image2.jpeg demo/image4.jpeg False 0.9917208 OpenFace   opencv
    ## 3  demo/image2.jpeg demo/image3.jpeg False 1.0273918 OpenFace   opencv
    ## 4  demo/image2.jpeg  demo/image5.jpg False 0.9787431 OpenFace   opencv
    ## 5  demo/image1.jpeg demo/image4.jpeg  True 0.5256947 OpenFace   opencv
    ## 6  demo/image1.jpeg demo/image3.jpeg False 0.7789021 OpenFace   opencv
    ## 7  demo/image1.jpeg  demo/image5.jpg  True 0.4791716 OpenFace   opencv
    ## 8  demo/image4.jpeg demo/image3.jpeg False 1.1320910 OpenFace   opencv
    ## 9  demo/image4.jpeg  demo/image5.jpg  True 0.4099046 OpenFace   opencv
    ## 10 demo/image3.jpeg  demo/image5.jpg False 1.1418931 OpenFace   opencv

We see that image4 and image5 are treated as the same person. Depending
on your research objective, you might want to remove duplicate faces.

### 1.2. Generate Predictions

#### 1.2.1 Use CNN Model

We provide a python function `img_pred` (in `img_pred.py`) that allows
for users to predict ethnorace for each image using `deepface` developed
by [Sefik Serengil](https://github.com/serengil/deepface) that uses a
pre-trained CNN model
[VGG-Face](https://sefiks.com/2018/08/06/deep-face-recognition-with-keras/)
to predict ethnorace from an image. This package is available in python
and is fairly straightforward to use.

The `img_pred` function takes the following arguments:

- `imgfolder` (string): Name of the folder where the images are saved.
- `detector` (string): Detector backend for facial recognition. Choose
  from “opencv” (default), “retinaface”, “mtcnn”, “ssd”, “dlib”,
  “mediapipe” or “yolov8”.
- `out_csv` (boolean): Save the result as a csv file to a `imgfolder`.
  Default is TRUE.

``` python
img_pred(imgfolder = 'demo', detector = "opencv", out_csv = True)
```

    ##        asian     indian       black      white middle.eastern latino.hispanic
    ## 1  0.7306429  0.7287341 94.75514889  0.1573047     0.16038986        3.467783
    ## 2  7.0497677  1.4282474  0.56429561 61.4359796    15.88970423       13.632008
    ## 3 13.1156013 13.9323832 24.71523282 10.1571959     7.96273669       30.116857
    ## 4  5.7343315  3.8855933  2.02840026 42.7331328    18.24297458       27.375564
    ## 5 66.7891046  4.2102424 26.45902055  0.1515177     0.02792631        2.362195
    ## 6  1.3253208  0.9619726  0.09824114 61.1619294    20.93248963       15.520045
    ##     dominant_race               fn detector
    ## 1           black demo/image2.jpeg   opencv
    ## 2           white demo/image2.jpeg   opencv
    ## 3 latino hispanic demo/image1.jpeg   opencv
    ## 4           white demo/image1.jpeg   opencv
    ## 5           asian demo/image1.jpeg   opencv
    ## 6           white demo/image1.jpeg   opencv

Not that there are many many different CNN models available. Our
function is merely a demonstration using one of these models.

## 2. Surname-Based Prediction

Use `wru` package created by [Imai and Khanna
(2016)](https://imai.fas.harvard.edu/research/race.html) to generate
surname-based predictions. For example, using the `demo` dataset,

``` r
library(wru)
load('demo/demo_data.rdata')
demo <- demo[,names(demo)[!grepl('bayes', names(demo))]]
demo1 <- predict_race(voter.file = demo, surname.only = T)
demo1 <- demo1 %>% rename_at(vars(matches('^pred\\.')), ~ gsub('pred', 'bayes', .x))
```

## 3. Run multinomial regression

Run multinomial model using prediction probability distributions from
image as well as bayesian methods. For example, using the `demo`
dataset:

``` r
library(nnet)
cov <- names(demo1)[which(grepl('bayes|image', names(demo1)))]
eq  <- as.formula(paste('race ~ ', paste(cov, collapse = ' + ')))
print(eq)
```

    ## race ~ image.asian + image.indian + image.black + image.white + 
    ##     image.middle.eastern + image.latino.hispanic + bayes.whi + 
    ##     bayes.bla + bayes.his + bayes.asi + bayes.oth

``` r
mod <- multinom(eq, data = demo1)
```

    ## # weights:  52 (36 variable)
    ## initial  value 40.202536 
    ## iter  10 value 17.107391
    ## iter  20 value 7.841235
    ## iter  30 value 4.328976
    ## iter  40 value 1.717132
    ## iter  50 value 0.001449
    ## final  value 0.000067 
    ## converged

``` r
demo1$pred_race = predict(mod, newdata = demo1, "class")
demo2 <- cbind(demo1, predict(mod, newdata = demo1, "probs"))
head(demo2[,c(6, 20:24)])
```

    ##           fullname pred_race         asian         black     caucasian
    ## 1   shannon hardin     black  0.000000e+00  1.000000e+00 3.375924e-173
    ## 2 michael flaherty caucasian  0.000000e+00 1.011087e-134  1.000000e+00
    ## 3    john bartlett caucasian 2.032804e-101  9.270391e-27  1.000000e+00
    ## 4    katrina foley caucasian  0.000000e+00 5.201446e-125  1.000000e+00
    ## 5  greg brockhouse caucasian  0.000000e+00 2.528776e-152  1.000000e+00
    ## 6    gina driscoll caucasian 1.705588e-314  5.687802e-80  1.000000e+00
    ##        hispanic
    ## 1 3.636397e-145
    ## 2  0.000000e+00
    ## 3 2.958698e-244
    ## 4  0.000000e+00
    ## 5  0.000000e+00
    ## 6  0.000000e+00

Users may employ fancier machine learning algorithms (but our validation
finds that this simple multinomial model performs just as well).
