 #!/bin/bash

 echo "Start pdflatex on main.txt"

 # First
pdflatex main.tex

 # second
 pdflatex main.tex

 rm main.dvi
 rm main.log
 rm main.aux
 rm main.lof
 rm main.lot
 rm main.out
 rm main.toc
 rm main.run.xml
 rm main.bcf:wq
 