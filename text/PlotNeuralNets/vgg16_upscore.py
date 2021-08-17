
import sys
import os
sys.path.append(os.path.abspath('/home/thesis/PlotNeuralNets/'))
# print(sys.path)
from pycore.tikzeng import *
from pycore.blocks import *

OFFSET_X = 2
OFFSET_Y = 5

# defined your arch
arch = [

    to_head('.'), 
    to_cor(),
    to_begin(),
#input
    to_input('/home/thesis/PlotNeuralNets/images/input.png' , width=10.0, height=10.0, name='input'),
    
    #conv1
    to_ConvConvRelu( name='cr1', s_filer=352, n_filer=(64,64), offset="(0,0,0)", to="(0,0,0)", width=(2,2), height=40, depth=40, caption="conv1"),
    to_Pool(name="p1", offset="(0,0,0)", to="(cr1-east)", width=1, height=35, depth=35, opacity=0.5),
    
    #conv2
    to_ConvConvRelu( name='cr2', s_filer=128, n_filer=(128,128), offset=f"({OFFSET_X},0,0)", to="(p1-east)", width=(4,4), height=35, depth=35, caption="conv2"),
    to_Pool(name="p2", offset="(0,0,0)", to="(cr2-east)", width=1, height=30, depth=30, opacity=0.5),
    
    #conv3
    to_ConvConvRelu( name='cr3', s_filer=64, n_filer=(256,256,256), offset=f"({OFFSET_X},0,0)", to="(p2-east)", width=(4,4,4), height=30, depth=30, caption="conv3"),
    to_Pool(name="p3", offset="(0,0,0)", to="(cr3-east)", width=1, height=23, depth=23, opacity=0.5),
    
    #conv4
    to_ConvConvRelu( name='cr4', s_filer=32, n_filer=(512,512,512), offset=f"({OFFSET_X},0,0)", to="(p3-east)", width=(4,4,4), height=23, depth=23, caption="conv4"),
    to_Pool(name="p4", offset="(0,0,0)", to="(cr4-east)", width=1, height=15, depth=15, opacity=0.5),
    
    #conv5
    to_ConvConvRelu( name='cr5', s_filer=16, n_filer=(512,512,512), offset=f"({OFFSET_X},0,0)", to="(p4-east)", width=(4,4,4), height=15, depth=15, caption="conv5"),
    to_Pool(name="p5", offset="(0,0,0)", to="(cr5-east)", width=1, height=10, depth=10, opacity=0.5),

    to_ConvConvRelu( name='fc7', s_filer=16, n_filer=(4096, 4096), offset=f"({OFFSET_X},0,0)", to="(p5-east)", width=(6, 6), height=15, depth=15, caption="conv6"),

    to_Conv( name = 'score', s_filer=16, n_filer=6, offset=f"({OFFSET_X},0,0)", to='(fc7-east)', width=1, height=15, depth=15, caption="features" ),
    


    to_ConvRes( name='res3',   offset=f"({OFFSET_X },0,0)", to="(score-east)",    s_filer=32, n_filer=(6), width=(1), height=23, depth=23, opacity=0.5 ),
    to_Conv( name = 'up3', s_filer=32, n_filer=6, offset=f"(0,{OFFSET_Y},0)", to='(res3-west)', width=1, height=23, depth=23 ),
    
    to_Sum("s3", to="(res3-east)", offset=f"({OFFSET_X},0,0)"),

    to_ConvRes( name='res4',   offset=f"({OFFSET_X},0,0)", to="(s3-east)",    s_filer=64, n_filer=(6), width=(1), height=30, depth=30, opacity=0.5 ),
    to_Conv( name = 'up4', s_filer=64, n_filer=6, offset=f"(0,{OFFSET_Y},0)", to='(res4-west)', width=1, height=30, depth=30 ),
    
    to_Sum("s4", to="(res4-east)", offset=f"({OFFSET_X},0,0)"),

    to_ConvRes( name='res5',   offset=f"({OFFSET_X},0,0)", to="(s4-east)",    s_filer=352, n_filer=(6), width=(1), height=40, depth=40, opacity=0.5 ),
    
    
    #connections
    to_connection( "p1", "cr2"),
    to_connection( "p2", "cr3"),
    to_connection( "p3", "cr4"),
    to_connection( "p4", "cr5"),
    to_connection( "p5", "fc7"),
    to_connection( "fc7", "score"),
    to_connection( "score", "res3"),
    to_connection( "res3", "s3"),
    to_connection( "s3", "res4"),
    to_connection( "res4", "s4"),
    to_connection( "s4", "res5"),

    to_end() 
    ]

def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex' )

if __name__ == '__main__':
    main()
