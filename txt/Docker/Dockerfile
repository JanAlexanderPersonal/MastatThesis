FROM ubuntu:18.04

ENV TZ=Europe/Brussels
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && \
    apt-get upgrade -y

RUN apt-get install -y apt-utils
RUN apt-get install -y --no-install-recommends texlive-latex-recommended 
RUN apt-get install -y texlive-latex-extra 
RUN apt-get install -y texlive-fonts-recommended 
RUN apt-get install -y texlive-fonts-extra 
RUN apt-get install -y cm-super
RUN apt-get install lmodern

# RUN apt-get install texlive-lang-all

WORKDIR /home/thesis
CMD ["sh", "latex_thesis.sh"]