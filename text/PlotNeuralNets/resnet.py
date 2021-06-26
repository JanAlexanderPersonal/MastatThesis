import sys
import os
sys.path.append(os.path.abspath('/home/thesis/PlotNeuralNets/'))
# print(sys.path)
from pycore.tikzeng import *
from pycore.blocks import *

OFFSET_X = 10
OFFSET_Y = -10

# Define architecture:

arch = [
    to_head('.'), 
    to_cor(),
    to_begin(),
#input
    to_input('/home/thesis/PlotNeuralNets/images/input.png' , width=10.0, height=10.0, name='input'),

    to_Conv( name='conv1', s_filer=352, n_filer=16, offset="(0,0,0)", to="(0,0,0)", width=2, height=40, depth=40, caption="conv1"),
    
    to_ConvRes( name='L1_1', s_filer=352, n_filer=16, offset=f"({OFFSET_X},0,0)", to="(conv1-east)", width=2, height=40, depth=40, caption="3x3"),
    to_Conv(name = 'L1_2', s_filer=352, n_filer=16, offset=f"(0,0,0)", to='(L1_1-east)', width=2, height=40, depth=40, caption={"","BN"} ),

    to_Sum('sum1', offset=f"({OFFSET_X},0,0)", to='(L1_2-east)', radius=1.5, opacity=0.6),
    

    to_connection( "conv1", "L1_1"),


    to_end() 
]

def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex' )

if __name__ == '__main__':
    main()
