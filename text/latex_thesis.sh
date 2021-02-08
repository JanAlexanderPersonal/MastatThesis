#!/bin/bash
# run the latest pdf container (interactive)

latex main.tex
bibtex main.tex
pdflatex main.tex
makeglossaries main
biber main
pdflatex main.tex
pdflatex main.tex

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