#!/bin/bash
# run the latest pdf container (interactive)

# rm /home/thesis/PlotNeuralNets/*.tex
rm /home/thesis/PlotNeuralNets/*.aux

echo "Maak images netwerken"
#python3.8 /home/thesis/PlotNeuralNets/unet.py > /home/thesis/PlotNeuralNets/unet.tex
#python3.8 /home/thesis/PlotNeuralNets/vgg16_upscore.py > /home/thesis/PlotNeuralNets/vgg16.tex

pdflatex /home/thesis/PlotNeuralNets/unet.tex
pdflatex /home/thesis/PlotNeuralNets/vgg16_upscore.tex
pdflatex /home/thesis/PlotNeuralNets/resnet.tex

echo "Make main document"
pdflatex main.tex
bibtex main.tex
pdflatex main.tex
makeglossaries main
biber main
pdflatex main.tex
bibtex main.tex
biber main
bibtex main.tex
biber main
pdflatex main.tex

echo "remove extra log documents"
rm *.out
rm *.toc
rm *.log
rm *.fls
rm *.dvi
rm *.blg
rm *.acn
rm *.alg
rm *.glg
rm *.gls
rm *.ist
rm *.xdy
rm *.glo
rm *.acr
rm *.fdb_latexmk
rm *.bbl
rm *.aux
rm *.bcf
rm *.xml
rm *.fls
rm *.bib.bak
rm *.fdb_latexmk

rm /home/thesis/PlotNeuralNets/*.fls
rm /home/thesis/PlotNeuralNets/*.latexmk
rm /home/thesis/PlotNeuralNets/*.fdb_latexmk
rm /home/thesis/PlotNeuralNets/*.log