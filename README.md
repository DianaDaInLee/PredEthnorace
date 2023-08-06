Step-by-Step Guidelines to Implement Hybrid Ethnoracial Prediction
================
Diana Da In Lee (<dl2860@columbia.edu>)
Last Updated: 06 August, 2023

This guideline walks through how to implement the proposed hybrid
approach to ethnoracial prediction in Lee, Diana, and Yamil Velez.
“[Measuring descriptive representation at scale: Methods for predicting
the race and ethnicity of public officials.](https://osf.io/tpsv6/)”
(2023+).

We will use the US local elections dataset compiled by [de-Benedictis
Kessner et al (2023)](https://dash.harvard.edu/handle/1/37373139). We’ve
sampled 100 elections from the main dataset with 30% of the names with
verified ethnoracial information:

    ## # A tibble: 6 × 7
    ##   state city            year firstname surname    fullname         race     
    ##   <chr> <chr>          <dbl> <chr>     <chr>      <chr>            <chr>    
    ## 1 OH    columbus        2021 shannon   hardin     shannon hardin   <NA>     
    ## 2 MA    boston          2021 michael   flaherty   michael flaherty <NA>     
    ## 3 NJ    passaic         2021 john      bartlett   john bartlett    caucasian
    ## 4 CA    orange          2021 katrina   foley      katrina foley    <NA>     
    ## 5 TX    san antonio     2021 greg      brockhouse greg brockhouse  <NA>     
    ## 6 FL    st. petersburg  2021 gina      driscoll   gina driscoll    <NA>

## 1. Image-Based Prediction

We first want to collect images associated with each name and make
predictions based on the images. We provide two python functions
`img_search` and `img_pred` to do these steps. For users who don’t have
python installed in their computer, we also provide a Google Colab
demonstration here:
[PredEthnorace_Demo.ipynb](https://github.com/DianaDaInLee/PredEthnorace/blob/main/PredEthnorace_Demo.ipynb).

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
pred <- read.csv('demo/img_pred.csv') %>%
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

    FALSE   state           city year firstname    surname         fullname      race
    FALSE 1    OH       columbus 2021   shannon     hardin   shannon hardin      <NA>
    FALSE 2    MA         boston 2021   michael   flaherty michael flaherty      <NA>
    FALSE 3    NJ        passaic 2021      john   bartlett    john bartlett caucasian
    FALSE 4    CA         orange 2021   katrina      foley    katrina foley      <NA>
    FALSE 5    TX    san antonio 2021      greg brockhouse  greg brockhouse      <NA>
    FALSE 6    FL st. petersburg 2021      gina   driscoll    gina driscoll      <NA>
    FALSE            keyword image.asian image.indian image.black image.white
    FALSE 1   shannon_hardin          NA           NA          NA          NA
    FALSE 2 michael_flaherty          NA           NA          NA          NA
    FALSE 3    john_bartlett          NA           NA          NA          NA
    FALSE 4    katrina_foley          NA           NA          NA          NA
    FALSE 5  greg_brockhouse          NA           NA          NA          NA
    FALSE 6    gina_driscoll          NA           NA          NA          NA
    FALSE   image.middle.eastern image.latino.hispanic bayes.whi   bayes.bla  bayes.his
    FALSE 1                   NA                    NA 0.7252565 0.160995223 0.02512640
    FALSE 2                   NA                    NA 0.9316636 0.004681462 0.02188686
    FALSE 3                   NA                    NA 0.8648793 0.038173052 0.03105697
    FALSE 4                   NA                    NA 0.8879568 0.034392678 0.02535202
    FALSE 5                   NA                    NA 0.8999722 0.000000000 0.02913244
    FALSE 6                   NA                    NA 0.9131907 0.016200660 0.02926635
    FALSE     bayes.asi  bayes.oth
    FALSE 1 0.008044030 0.08057786
    FALSE 2 0.009206857 0.03256120
    FALSE 3 0.009371682 0.05651902
    FALSE 4 0.010455758 0.04184277
    FALSE 5 0.005769053 0.06512630
    FALSE 6 0.008770564 0.03257172

## 3. Run multinomial logistic regression

``` r
library(nnet)
#mod <- multinom(race ~ img.whi + img.bla + img.his + img.asi + bayes.whi + bayes.bla + bayes.his + bayes.asi + bayes.oth, data = demo)
#demo$pred_race = predict(mod, newdata = demo, "class")
```

Users are free to employ fancier machine learning algorithms (but our
validation finds that this simple multinomial model performs just as
well).
