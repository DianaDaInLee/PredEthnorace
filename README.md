Step-by-Step Guidelines to Implement Hybrid Ethnoracial Prediction
================
Diana Da In Lee (<dl2860@columbia.edu>)
Last Updated: 06 August, 2023

This guideline walks through how to implement the proposed hybrid
approach to ethnoracial prediction in Lee, Diana, and Yamil Velez.
“[Measuring descriptive representation at scale: Methods for predicting
the race and ethnicity of public officials.](https://osf.io/tpsv6/)”
(2023+).

We will use the following example data `demo` on 100 US city councilor
names with 30% of the names have verified ethnoracial information:

    ##    state         city year       office firstname       surname
    ## 6     NY  Cheektowaga 2019 City Council   richard      rusiniak
    ## 8     NY Mount Vernon 2019 City Council   derrick      thompson
    ## 12    NY New Rochelle 2019 City Council    yadira ramos-herbert
    ## 13    NY New Rochelle 2019 City Council    albert     tarantino
    ## 29    IL      Chicago 2019 City Council     brian       hopkins
    ## 39    IL      Chicago 2019 City Council     jason         ervin
    ##                fullname race
    ## 6   richard j. rusiniak <NA>
    ## 8      derrick thompson <NA>
    ## 12 yadira ramos-herbert <NA>
    ## 13     albert tarantino <NA>
    ## 29        brian hopkins <NA>
    ## 39          jason ervin <NA>

## 1. Image-Based Prediction

We first want to collect images associated with each name and make
predictions based on the images. We provide two python functions
`img_search` and `img_pred` to do these steps. For users who don’t have
python installed in their computer, we also provide a Google Colab
demonstration here:.

### 1.1. Collect Images

Collect profile images of the individuals in the dataset, using their
first and last names as a keyword. Users may use any public engine. We
provide a python function `img_search` (in `img_pred.py`) that allows
for users to collect images from [DuckDuckGo](https://duckduckgo.com).
The function automatically 1) removes images with no faces; 2) removes
images with multiple faces; and 3) de-dupes identical faces (remove one
of the images that are detected to be of a same person as the one in
another image).

The `img_search` function in takes the following arguments:

- `fullname` (string): First and last name to be used as a keyword for
  image search.
- `maxnum` (integer): Total number of images you want to collect per
  keyword.
- `imgfolder` (string): Name of the folder you want the images to be
  saved. It will automatically create the folder in your working
  directory.

Each image will be saved in jpg as \[`keyword`\]\_\[0 through
`maxnum`\].jpg. All images associated with each keyword will be stored
in its own folder named after the keyword (so make sure to dedup the
full names in your dataset!).

``` python
img_search(fullname = 'diana lee', maxnum = 3, imgfolder = 'demo')
glob.glob('demo/*/*')
```

``` python
['demo/diana_lee/diana_lee_0.jpg',
 'demo/diana_lee/diana_lee_1.jpg',
 'demo/diana_lee/diana_lee_2.jpg']
```

### 1.2. Generate Predictions

We provide a python function `img_pred` (in `img_pred.py`) that allows
for users to predict ethnorace for each image using `deepface` developed
by [Sefik Serengil](https://github.com/serengil/deepface) that uses a
pre-trained CNN model
[VGG-Face](https://sefiks.com/2018/08/06/deep-face-recognition-with-keras/)
to predict ethnorace from an image. This package is available in python
and is fairly straightforward to use.

The `img_pred` function takes the following arguments:

- `imgfolder` (string): Name of the folder where the images are saved.
  Should be the same as the folder name used in `img_search`.
- `subfolder` (string): Name of a particular subfolder (i.e., keyword)
  under `imgfolder` that you want predictions to be performed. Default
  is NULL, which will generate predictions for all images in all
  subfolders saved in `imgfolder`.
- `det` (string): face detector backend to opencv (default), retinaface,
  mtcnn, ssd, dlib, mediapipe or yolov8. See
  <https://github.com/serengil/deepface> for differences across
  different detectors.
- `out_csv` (boolean): Save the result as a csv file to a `imgfolder`.
  Default is TRUE.

``` python
pred = img_pred(imgfolder = 'demo', subfolder = 'diana_lee', det = "opencv", out_csv = True)
pred
```

Not that there are many many different CNN models available. Our
function is merely a demonstration using one of these models.

### 1.3 All at Once

For large dataset with many names, loop it!

``` python
# Python
import pyreadr
df = pyreadr.read_r('demo.rdata') 
demo = df["demo"]

# Collect Images
for index, row in demo.iterrows():
    print(row['fullname'])
    img_search(fullname = row['fullname'], maxnum = 3, imgfolder = 'demo')
    
# Generate Predictions
img_pred(imgfolder = 'demo', det = "opencv", out_csv = True)
```

``` r
# R
# Merge It with Raw Data
pred <- read.csv('data/img_pred.csv') %>%
  separate(fn, c('imgfolder', 'keyword', 'filenmae'), sep = '/') %>%
  select(asian:latino.hispanic, keyword) %>%
  rename_at(vars(asian:latino.hispanic), ~ paste0('image.', .x))
demo <- left_join(demo %>% mutate(keyword = tolower(gsub(" ", "_", fullname))), pred)
```

    ## Joining with `by = join_by(keyword)`

## 2. Surname-Based Prediction

This part is very simple. Use `wru` package created by [Imai and Khanna
(2016)](https://imai.fas.harvard.edu/research/race.html) to generate
surname-based predictions:

``` r
library(wru)
demo <- predict_race(voter.file = demo, surname.only = T)
demo <- demo %>% rename_at(vars(matches('^pred')), ~ gsub('pred', 'bayes', .x))
head(demo)
```

    FALSE    place_fips state         city year       office firstname       surname
    FALSE 1     3615000    NY  Cheektowaga 2019 City Council   richard      rusiniak
    FALSE 2     3649121    NY Mount Vernon 2019 City Council   derrick      thompson
    FALSE 92    3650617    NY New Rochelle 2019 City Council    yadira ramos-herbert
    FALSE 3     3650617    NY New Rochelle 2019 City Council    albert     tarantino
    FALSE 4     1714000    IL      Chicago 2019 City Council     brian       hopkins
    FALSE 5     1714000    IL      Chicago 2019 City Council     jason         ervin
    FALSE                fullname race              keyword image.asian image.indian
    FALSE 1   richard j. rusiniak <NA>  richard_j._rusiniak          NA           NA
    FALSE 2      derrick thompson <NA>     derrick_thompson          NA           NA
    FALSE 92 yadira ramos-herbert <NA> yadira_ramos-herbert          NA           NA
    FALSE 3      albert tarantino <NA>     albert_tarantino          NA           NA
    FALSE 4         brian hopkins <NA>        brian_hopkins          NA           NA
    FALSE 5           jason ervin <NA>          jason_ervin          NA           NA
    FALSE    image.black image.white image.middle.eastern image.latino.hispanic
    FALSE 1           NA          NA                   NA                    NA
    FALSE 2           NA          NA                   NA                    NA
    FALSE 92          NA          NA                   NA                    NA
    FALSE 3           NA          NA                   NA                    NA
    FALSE 4           NA          NA                   NA                    NA
    FALSE 5           NA          NA                   NA                    NA
    FALSE     bayes.whi   bayes.bla  bayes.his   bayes.asi  bayes.oth
    FALSE 1  0.95249360 0.000000000 0.01157469 0.013752710 0.02217901
    FALSE 2  0.64759552 0.235710952 0.02892281 0.007866069 0.07990464
    FALSE 92 0.05172637 0.008414011 0.86584734 0.057962502 0.01604978
    FALSE 3  0.90709588 0.003516968 0.05106396 0.005893921 0.03242927
    FALSE 4  0.68868050 0.208418805 0.02506819 0.006673289 0.07115921
    FALSE 5  0.53933798 0.355056712 0.02835687 0.007017104 0.07023134

## 3. Run multinomial logistic regression

``` r
library(nnet)
#mod <- multinom(race ~ img.whi + img.bla + img.his + img.asi + bayes.whi + bayes.bla + bayes.his + bayes.asi + bayes.oth, data = demo)
#demo$pred_race = predict(mod, newdata = demo, "class")
```

Users are free to employ fancier machine learning algorithms (but our
validation finds that this simple multinomial model performs just as
well).
