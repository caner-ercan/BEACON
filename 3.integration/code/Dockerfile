FROM rocker/geospatial:4.3.3

WORKDIR /tmp/

RUN wget -O /install_tidyverse.sh https://raw.githubusercontent.com/rocker-org/rocker-versioned2/master/scripts/install_tidyverse.sh
RUN chmod +x /install_tidyverse.sh
RUN /install_tidyverse.sh

RUN install2.r --error \
    ggpubr \
    furrr \
    future \
    spaa \
    ggpubr \
    pracma \
    abdiv \
    ggdendro \
    pheatmap \
    pROC \
    caret \
    glmnet \
    forestmodel \
    survminer 

WORKDIR /mnt/data/app/rstudio/
RUN mkdir -p /mnt/data/app/rstudio/
ENV R_PROFILE_USER=/mnt/data/app/rstudio/.Rprofile
RUN echo 'setwd("/mnt/data/app/rstudio/")' > /mnt/data/app/rstudio/.Rprofile && \
    echo 'options(defaultPackages=c(getOption("defaultPackages"), "base"))' >> /mnt/data/app/rstudio/.Rprofile
RUN chown -R rstudio:rstudio /mnt/data/app/rstudio/

RUN echo 'setwd("/mnt/data/app/rstudio/")' > /mnt/data/app/rstudio/.Rprofile && \
    echo '.Last <- function() { if(interactive()) save.image("/mnt/data/app/rstudio/.RData") }' >> /mnt/data/app/rstudio/.Rprofile
    