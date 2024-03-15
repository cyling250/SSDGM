wipe
model basic -ndm 3 -ndf 6

node 1   0.00E+00      0.000         0.000            
node 2   5.00E+00      0.000         0.000            
node 3   1.00E+01      0.000         0.000            
node 4   1.50E+01      0.000         0.000            
node 5   2.00E+01      0.000         0.000            
node 6   2.50E+01      0.000         0.000            
node 7   3.00E+01      0.000         0.000            
node 8   3.50E+01      0.000         0.000            
node 9   4.00E+01      0.000         0.000            
node 10  4.50E+01      0.000         0.000            
node 11  5.00E+01      0.000         0.000            
node 12  5.50E+01      0.000         0.000            
node 13  6.00E+01      0.000         0.000            
node 14  6.50E+01      0.000         0.000            
node 15  7.00E+01      0.000         0.000            
node 16  7.50E+01      0.000         0.000            
node 17  8.00E+01      0.000         0.000            
node 18  8.50E+01      0.000         0.000            
node 19  9.00E+01      0.000         0.000            
node 20  9.50E+01      0.000         0.000            
node 21  1.00E+02      0.000         0.000            

node 22  2.00E+01      2.50E+00       -8.05E+00       
node 23  2.00E+01      2.50E+00       -4.55E+00       
node 24  2.00E+01      2.50E+00       -1.05E+00       
node 25  2.00E+01      2.50E+00       -1.05E+00
node 26  2.00E+01      2.50E+00        0.00E+00       
                                                      
node 27  2.00E+01     -2.50E+00       -8.05E+00       
node 28  2.00E+01     -2.50E+00       -4.55E+00       
node 29  2.00E+01     -2.50E+00       -1.05E+00       
node 30  2.00E+01     -2.50E+00       -1.05E+00
node 31  2.00E+01     -2.50E+00        0.00E+00       
                                                      
node 32  4.00E+01      0.00E+00       -9.05E+00       
node 33  4.00E+01      0.00E+00       -5.95E+00       
node 34  4.00E+01      0.00E+00       -2.85E+00       
node 35  4.00E+01      0.00E+00       -1.95E+00       
node 36  4.00E+01      0.00E+00       -1.05E+00       
node 37  4.00E+01      0.00E+00       -1.05E+00       
                                                      
node 43  6.00E+01      0.00E+00       -9.05E+00       
node 44  6.00E+01      0.00E+00       -5.95E+00       
node 45  6.00E+01      0.00E+00       -2.85E+00       
node 46  6.00E+01      0.00E+00       -1.95E+00       
node 47  6.00E+01      0.00E+00       -1.05E+00       
node 48  6.00E+01      0.00E+00       -1.05E+00       
                                                      
node 54  8.00E+01      2.50E+00       -8.05E+00       
node 55  8.00E+01      2.50E+00       -4.55E+00       
node 56  8.00E+01      2.50E+00       -1.05E+00       
node 57  8.00E+01      2.50E+00       -1.05E+00
node 58  8.00E+01      2.50E+00        0.00E+00       
                                                      
node 59  8.00E+01     -2.50E+00       -8.05E+00       
node 60  8.00E+01     -2.50E+00       -4.55E+00       
node 61  8.00E+01     -2.50E+00       -1.05E+00       
node 62  8.00E+01     -2.50E+00       -1.05E+00
node 63  8.00E+01     -2.50E+00        0.00E+00       
set l 1
# nodes of the link element
# ???
node 64  -$l            0               0       
node 65  101         0               0

# nodes of the zero-length element
node 66  2.00E+01      2.50E+00       -8.05E+00  
node 67  2.00E+01     -2.50E+00       -8.05E+00  
node 68  4.00E+01      0.00E+00       -9.05E+00
node 69  6.00E+01      0.00E+00       -9.05E+00  
node 70  8.00E+01      2.50E+00       -8.05E+00     
node 71  8.00E+01     -2.50E+00       -8.05E+00

                                                             
mass 1   2.9895E+04 2.9895E+04 2.9895E+04 0.000E+000 0.000E+000 0.000E+000
mass 2   5.9791E+04 5.9791E+04 5.9791E+04 0.000E+000 0.000E+000 0.000E+000
mass 3   5.9791E+04 5.9791E+04 5.9791E+04 0.000E+000 0.000E+000 0.000E+000
mass 4   5.9791E+04 5.9791E+04 5.9791E+04 0.000E+000 0.000E+000 0.000E+000
mass 5   5.9791E+04 5.9791E+04 5.9791E+04 0.000E+000 0.000E+000 0.000E+000
mass 6   5.9791E+04 5.9791E+04 5.9791E+04 0.000E+000 0.000E+000 0.000E+000
mass 7   5.9791E+04 5.9791E+04 5.9791E+04 0.000E+000 0.000E+000 0.000E+000
mass 8   5.9791E+04 5.9791E+04 5.9791E+04 0.000E+000 0.000E+000 0.000E+000
mass 9   5.9791E+04 5.9791E+04 5.9791E+04 0.000E+000 0.000E+000 0.000E+000
mass 10  5.9791E+04 5.9791E+04 5.9791E+04 0.000E+000 0.000E+000 0.000E+000
mass 11  5.9791E+04 5.9791E+04 5.9791E+04 0.000E+000 0.000E+000 0.000E+000
mass 12  5.9791E+04 5.9791E+04 5.9791E+04 0.000E+000 0.000E+000 0.000E+000
mass 13  5.9791E+04 5.9791E+04 5.9791E+04 0.000E+000 0.000E+000 0.000E+000
mass 14  5.9791E+04 5.9791E+04 5.9791E+04 0.000E+000 0.000E+000 0.000E+000
mass 15  5.9791E+04 5.9791E+04 5.9791E+04 0.000E+000 0.000E+000 0.000E+000
mass 16  5.9791E+04 5.9791E+04 5.9791E+04 0.000E+000 0.000E+000 0.000E+000
mass 17  5.9791E+04 5.9791E+04 5.9791E+04 0.000E+000 0.000E+000 0.000E+000
mass 18  5.9791E+04 5.9791E+04 5.9791E+04 0.000E+000 0.000E+000 0.000E+000
mass 19  5.9791E+04 5.9791E+04 5.9791E+04 0.000E+000 0.000E+000 0.000E+000
mass 20  5.9791E+04 5.9791E+04 5.9791E+04 0.000E+000 0.000E+000 0.000E+000
mass 21  2.9895E+04 2.9895E+04 2.9895E+04 0.000E+000 0.000E+000 0.000E+000
mass 22  4.7521E+03 4.7521E+03 4.7521E+03 0.000E+000 0.000E+000 0.000E+000
mass 23  9.5041E+03 9.5041E+03 9.5041E+03 0.000E+000 0.000E+000 0.000E+000
mass 24  4.7521E+03 4.7521E+03 4.7521E+03 0.000E+000 0.000E+000 0.000E+000
mass 27  4.7521E+03 4.7521E+03 4.7521E+03 0.000E+000 0.000E+000 0.000E+000
mass 28  9.5041E+03 9.5041E+03 9.5041E+03 0.000E+000 0.000E+000 0.000E+000
mass 29  4.7521E+03 4.7521E+03 4.7521E+03 0.000E+000 0.000E+000 0.000E+000
mass 32  6.5765E+03 6.5765E+03 6.5765E+03 0.000E+000 0.000E+000 0.000E+000
mass 33  1.3153E+04 1.3153E+04 1.3153E+04 0.000E+000 0.000E+000 0.000E+000
mass 34  1.1655E+04 1.1655E+04 1.1655E+04 0.000E+000 0.000E+000 0.000E+000
mass 35  1.3938E+04 1.3938E+04 1.3938E+04 0.000E+000 0.000E+000 0.000E+000
mass 36  7.3471E+03 7.3471E+03 7.3471E+03 0.000E+000 0.000E+000 0.000E+000
mass 43  6.5765E+03 6.5765E+03 6.5765E+03 0.000E+000 0.000E+000 0.000E+000
mass 44  1.3153E+04 1.3153E+04 1.3153E+04 0.000E+000 0.000E+000 0.000E+000
mass 45  1.1655E+04 1.1655E+04 1.1655E+04 0.000E+000 0.000E+000 0.000E+000
mass 46  1.3938E+04 1.3938E+04 1.3938E+04 0.000E+000 0.000E+000 0.000E+000
mass 47  7.3471E+03 7.3471E+03 7.3471E+03 0.000E+000 0.000E+000 0.000E+000
mass 54  4.7521E+03 4.7521E+03 4.7521E+03 0.000E+000 0.000E+000 0.000E+000
mass 55  9.5041E+03 9.5041E+03 9.5041E+03 0.000E+000 0.000E+000 0.000E+000
mass 56  4.7521E+03 4.7521E+03 4.7521E+03 0.000E+000 0.000E+000 0.000E+000
mass 59  4.7521E+03 4.7521E+03 4.7521E+03 0.000E+000 0.000E+000 0.000E+000
mass 60  9.5041E+03 9.5041E+03 9.5041E+03 0.000E+000 0.000E+000 0.000E+000
mass 61  4.7521E+03 4.7521E+03 4.7521E+03 0.000E+000 0.000E+000 0.000E+000
 
       
fix 1  0 1 1 1 1 1; 
fix 21 0 1 1 1 1 1; 
fix 66 1 1 1 1 1 1; 
fix 67 1 1 1 1 1 1; 
fix 68 1 1 1 1 1 1; 
fix 69 1 1 1 1 1 1; 
fix 70 1 1 1 1 1 1; 
fix 71 1 1 1 1 1 1; 
fix 64 1 1 1 1 1 1; 
fix 65 1 1 1 1 1 1; 

equalDOF 24 25 3 4 5 6
equalDOF 29 30 3 4 5 6
equalDOF 36 37 3 4 5 6
equalDOF 47 48 3 4 5 6
equalDOF 56 57 3 4 5 6
equalDOF 61 62 3 4 5 6
 
                    
set gap 0.64
uniaxialMaterial Concrete01 1 -3.285E+07 -0.00363 -1.911E+07 -0.00693      
uniaxialMaterial Concrete01 2 -3.285E+07 -0.00363 -1.911E+07 -0.00693     
uniaxialMaterial Elastic 3 1.20E+021 
# uniaxialMaterial Steel01 4 3.95E+04 1.1218E+07   0.07                   
# uniaxialMaterial Steel01 5 7.9E+04 2.2436E+07   0.07

uniaxialMaterial Steel01 4 4E+04 9.55E+06   0.0764                   
uniaxialMaterial Steel01 5 2.03E+05 1.897E+07   0.077

uniaxialMaterial Steel01 8 6E+04 2E+10   0.00000001
uniaxialMaterial ElasticPP 6 2E+9 1E-4
uniaxialMaterial Eblock 12 2E+16 0.06
uniaxialMaterial Impact02Material 14 2.33E+8 8E+7 -0.0025 -0.03

uniaxialMaterial Series 11 4 8
uniaxialMaterial Series 10 5 6
uniaxialMaterial  Parallel 13 11 12

# uniaxialMaterial LRBFatigue 4 1.4E5 7E7  9.36E5  0.0014  0.1375  0.015  0.0575 0.08 0.005 0.1
# uniaxialMaterial LRBFatigue 5 1.4E5 7E7  9.36E5  0.0014  0.1375  0.015  0.0575 0.08 0.005 0.1

#uniaxialMaterial SelfCentering 4 5.608243E+6 1.5267857E+6 3.028451E+5 1
#uniaxialMaterial SelfCentering 5 5.608243E+6 1.5267857E+6 3.028451E+5 1

uniaxialMaterial Concrete01 7 -1.7085E+07 -0.0011 -1.452E+07 -0.004       
uniaxialMaterial Steel01 9 3.8165E+08 2.00E+011 0.01                      
#uniaxialMaterial SelfCentering 9 5.608243E+6 1.5267857E+6 3.028451E+5 1
                      

uniaxialMaterial Elastic 201  1.0179E+17
uniaxialMaterial Elastic 301  1.0179E+17
uniaxialMaterial Elastic 401  2.0358E+17
uniaxialMaterial Elastic 202  1.5904E+17
uniaxialMaterial Elastic 302  1.5904E+17
uniaxialMaterial Elastic 402  4.9701E+17

##section R=1.200
section Fiber 1 -GJ 0 {
#patch circ 1 7 4 0 0 0 0.565 0 360
patch circ 7 7 2 0 0 0.565 0.600 0 360
layer circ 9 28 4.909E-04 0 0 0.530 0 360

}
##section R=1.500
section Fiber 2 -GJ 0 {
patch circ 2 5 4 0 0 0 0.710 0 360
patch circ 7 5 2 0 0 0.710 0.750 0 360
layer circ 9 40 4.909E-04 0 0 0.670 0 360
}

section Aggregator 1001 201 Vy 301 Vz 401 T -section 1
section Aggregator 1002 202 Vy 302 Vz 402 T -section 2

  
geomTransf Linear 1   0.000 0.000  1.000 
geomTransf Linear 2   0.000 0.000  1.000 
geomTransf Linear 3   0.000 0.000  1.000 
geomTransf Linear 4   0.000 0.000  1.000 
geomTransf Linear 5   0.000 0.000  1.000 
geomTransf Linear 6   0.000 0.000  1.000 
geomTransf Linear 7   0.000 0.000  1.000 
geomTransf Linear 8   0.000 0.000  1.000 
geomTransf Linear 9   0.000 0.000  1.000 
geomTransf Linear 10  0.000 0.000  1.000 
geomTransf Linear 11  0.000 0.000  1.000 
geomTransf Linear 12  0.000 0.000  1.000 
geomTransf Linear 13  0.000 0.000  1.000 
geomTransf Linear 14  0.000 0.000  1.000 
geomTransf Linear 15  0.000 0.000  1.000 
geomTransf Linear 16  0.000 0.000  1.000 
geomTransf Linear 17  0.000 0.000  1.000 
geomTransf Linear 18  0.000 0.000  1.000 
geomTransf Linear 19  0.000 0.000  1.000 
geomTransf Linear 20  0.000 0.000  1.000 
geomTransf Linear 21  0.000 -1.000  0.000
geomTransf Linear 22  0.000 -1.000  0.000
geomTransf Linear 25  0.000 -1.000  0.000
geomTransf Linear 26  0.000 -1.000  0.000
geomTransf Linear 29  0.000 -1.000  0.000
geomTransf Linear 30  0.000 -1.000  0.000
geomTransf Linear 31  0.000 -1.000  0.000
geomTransf Linear 32  0.000 -1.000  0.000
geomTransf Linear 40  0.000 -1.000  0.000
geomTransf Linear 41  0.000 -1.000  0.000
geomTransf Linear 42  0.000 -1.000  0.000
geomTransf Linear 43  0.000 -1.000  0.000
geomTransf Linear 50  0.000 -1.000  0.000
geomTransf Linear 51  0.000 -1.000  0.000
geomTransf Linear 54  0.000 -1.000  0.000
geomTransf Linear 55  0.000 -1.000  0.000

geomTransf Linear 64 0.000  0.000  1.000 
geomTransf Linear 65 0.000 -1.000  0.000
geomTransf Linear 66 0.000  0.000  1.000 
geomTransf Linear 67 0.000 -1.000  0.000
geomTransf Linear 68 0.000 -1.000  0.000
geomTransf Linear 69 0.000 -1.000  0.000
geomTransf Linear 70 0.000  0.000  1.000 
geomTransf Linear 71 0.000 -1.000  0.000
geomTransf Linear 72 0.000  0.000  1.000 
geomTransf Linear 73 0.000 -1.000  0.000
geomTransf Linear 74 0.000 0.000  1.000


element elasticBeamColumn 1   1  2   5.0441E+00 3.250E+010 1.354E+010 3.2912E+00 1.1445E+00 3.5162E+01  1 
element elasticBeamColumn 2   2  3   5.0441E+00 3.250E+010 1.354E+010 3.2912E+00 1.1445E+00 3.5162E+01  2 
element elasticBeamColumn 3   3  4   5.0441E+00 3.250E+010 1.354E+010 3.2912E+00 1.1445E+00 3.5162E+01  3 
element elasticBeamColumn 4   4  5   5.0441E+00 3.250E+010 1.354E+010 3.2912E+00 1.1445E+00 3.5162E+01  4 
element elasticBeamColumn 5   5  6   5.0441E+00 3.250E+010 1.354E+010 3.2912E+00 1.1445E+00 3.5162E+01  5 
element elasticBeamColumn 6   6  7   5.0441E+00 3.250E+010 1.354E+010 3.2912E+00 1.1445E+00 3.5162E+01  6 
element elasticBeamColumn 7   7  8   5.0441E+00 3.250E+010 1.354E+010 3.2912E+00 1.1445E+00 3.5162E+01  7 
element elasticBeamColumn 8   8  9   5.0441E+00 3.250E+010 1.354E+010 3.2912E+00 1.1445E+00 3.5162E+01  8 
element elasticBeamColumn 9   9  10  5.0441E+00 3.250E+010 1.354E+010 3.2912E+00 1.1445E+00 3.5162E+01  9 
element elasticBeamColumn 10  10 11  5.0441E+00 3.250E+010 1.354E+010 3.2912E+00 1.1445E+00 3.5162E+01  10
element elasticBeamColumn 11  11 12  5.0441E+00 3.250E+010 1.354E+010 3.2912E+00 1.1445E+00 3.5162E+01  11
element elasticBeamColumn 12  12 13  5.0441E+00 3.250E+010 1.354E+010 3.2912E+00 1.1445E+00 3.5162E+01  12
element elasticBeamColumn 13  13 14  5.0441E+00 3.250E+010 1.354E+010 3.2912E+00 1.1445E+00 3.5162E+01  13
element elasticBeamColumn 14  14 15  5.0441E+00 3.250E+010 1.354E+010 3.2912E+00 1.1445E+00 3.5162E+01  14
element elasticBeamColumn 15  15 16  5.0441E+00 3.250E+010 1.354E+010 3.2912E+00 1.1445E+00 3.5162E+01  15
element elasticBeamColumn 16  16 17  5.0441E+00 3.250E+010 1.354E+010 3.2912E+00 1.1445E+00 3.5162E+01  16
element elasticBeamColumn 17  17 18  5.0441E+00 3.250E+010 1.354E+010 3.2912E+00 1.1445E+00 3.5162E+01  17
element elasticBeamColumn 18  18 19  5.0441E+00 3.250E+010 1.354E+010 3.2912E+00 1.1445E+00 3.5162E+01  18
element elasticBeamColumn 19  19 20  5.0441E+00 3.250E+010 1.354E+010 3.2912E+00 1.1445E+00 3.5162E+01  19
element elasticBeamColumn 20  20 21  5.0441E+00 3.250E+010 1.354E+010 3.2912E+00 1.1445E+00 3.5162E+01  20
element dispBeamColumn 21  22 23  2  1001 21
element dispBeamColumn 22  23 24  2  1001 22
element dispBeamColumn 25  27 28  2  1001 25
element dispBeamColumn 26  28 29  2  1001 26
element dispBeamColumn 29  32 33  2  1002 29
element dispBeamColumn 30  33 34  2  1002 30
element elasticBeamColumn 31  34 35  5.4000E+00 3.000E+010 1.250E+010 5.7400E+00 6.5507E+00 2.2667E+00  31
element elasticBeamColumn 32  35 36  6.8000E+00 3.000E+010   1.250E+010 5.7400E+00 6.5507E+00 2.2667E+00  32
element dispBeamColumn 40  43 44 2  1002 40
element dispBeamColumn 41  44 45 2  1002 41
element elasticBeamColumn 42  45 46  5.4000E+00 3.000E+010 1.250E+010 5.7400E+00 6.5507E+00 2.2667E+00  42
element elasticBeamColumn 43  46 47  6.8000E+00 3.000E+010 1.250E+010 5.7400E+00 6.5507E+00 2.2667E+00  43
element dispBeamColumn 50  54 55 2  1001 50
element dispBeamColumn 51  55 56 2  1001 51
element dispBeamColumn 54  59 60 2  1001 54
element dispBeamColumn 55  60 61 2  1001 55

element elasticBeamColumn 64   5 26   5.0441E+00 3.250E+008 1.354E+008 3.2912E+00 1.1445E+00 3.5162E+01  64 
element elasticBeamColumn 65  25 26   5.4000E+00 3.000E+008 1.250E+008 5.7400E+00 6.5507E+00 2.2667E+00  65
element elasticBeamColumn 66   5 31   5.0441E+00 3.250E+008 1.354E+008 3.2912E+00 1.1445E+00 3.5162E+01  66 
element elasticBeamColumn 67  30 31   5.4000E+00 3.000E+008 1.250E+008 5.7400E+00 6.5507E+00 2.2667E+00  67
element elasticBeamColumn 68  37  9   5.4000E+00 3.000E+008 1.250E+008 5.7400E+00 6.5507E+00 2.2667E+00  68
element elasticBeamColumn 69  48 13   5.4000E+00 3.000E+008 1.250E+008 5.7400E+00 6.5507E+00 2.2667E+00  69
element elasticBeamColumn 70  17 58   5.0441E+00 3.250E+008 1.354E+008 3.2912E+00 1.1445E+00 3.5162E+01  70 
element elasticBeamColumn 71  57 58   5.4000E+00 3.000E+008 1.250E+008 5.7400E+00 6.5507E+00 2.2667E+00  71
element elasticBeamColumn 72  17 63   5.0441E+00 3.250E+008 1.354E+008 3.2912E+00 1.1445E+00 3.5162E+01  72 
element elasticBeamColumn 73  62 63   5.4000E+00 3.000E+008 1.250E+008 5.7400E+00 6.5507E+00 2.2667E+00  73

element zeroLength 58 24 25 -mat 4 4   -dir 1 2  
element zeroLength 59 29 30 -mat 4 4   -dir 1 2  
element zeroLength 60 36 37 -mat 5 5   -dir 1 2  
element zeroLength 61 47 48 -mat 5 5   -dir 1 2  
element zeroLength 62 56 57 -mat 4 4  -dir 1 2  
element zeroLength 63 61 62 -mat 4 4  -dir 1 2 

#                          tag ndI ndJ  secTag
element zeroLengthSection  76   66   22  1001
#                          tag ndI ndJ  secTag
element zeroLengthSection  77   67   27  1001
#                          tag ndI ndJ  secTag
element zeroLengthSection  78   68   32  1002
#                          tag ndI ndJ  secTag
element zeroLengthSection  79   69   43  1002
#                          tag ndI ndJ  secTag
element zeroLengthSection  80   70   54  1001
#                         tag ndI ndJ  secTag
element zeroLengthSection  81   71  59  1001


set xDamp 0.05;
set nEigenI 1;
set nEigenJ 2;
set lambdaN [eigen [expr $nEigenJ]];
set lambdaI [lindex $lambdaN [expr $nEigenI-1]];
set lambdaJ [lindex $lambdaN [expr $nEigenJ-1]];
set omegaI [expr pow($lambdaI,0.5)]; 
set omegaJ [expr pow($lambdaJ,0.5)];
set alphaM [expr $xDamp*(2*$omegaI*$omegaJ)/($omegaI+$omegaJ)]; 
set betaKcurr [expr 2.*$xDamp/($omegaI+$omegaJ)];   
rayleigh $alphaM $betaKcurr 0 0  
set T1 [expr 2*3.14159/$omegaI]
set T2 [expr 2*3.14159/$omegaJ]
puts $T1

set PGA 1.3;set WAVE 2;
set IDloadTag 1001;
set iGMfile "$num.txt";
set iGMdirection "1"; 
set iGMfact "9.8"; 
set dt 0.02;   
foreach GMdirection $iGMdirection GMfile $iGMfile GMfact $iGMfact { 
incr IDloadTag; 
set GMfatt [expr 1*$GMfact];  
set AccelSeries "Series -dt $dt -filePath $iGMfile -factor  $GMfatt";
pattern UniformExcitation  $IDloadTag  $GMdirection -accel  $AccelSeries; 
}   

# bearing
recorder Element -file result/Defobearing-$num.txt -time -ele 58 59 60 61 62 63 deformation;
# pier curvature
recorder Node -file result/gbeamdisplace-$num.txt -time -node 5 -dof 1 disp
recorder Node -file result/hpilebotty#1-$num.txt -time -node 22 27 -dof 1 2 3 4 5 6 disp
recorder Node -file result/hpilebotty#2-$num.txt -time -node  32 -dof 1 2 3 4 5 6 disp
recorder Node -file result/hpilebotty#3-$num.txt -time -node  43 -dof 1 2 3 4 5 6 disp
recorder Node -file result/hpilebotty#4-$num.txt -time -node  54 59 -dof 1 2 3 4 5 6 disp
# Top Displacement
recorder Node -file result/duntopdisplace#1-$num.txt -time -node 24 29 -dof 1 2 3 disp
recorder Node -file result/duntopdisplace#2-$num.txt -time -node 36 -dof 1 2 3 disp
recorder Node -file result/duntopdisplace#3-$num.txt -time -node 47 -dof 1 2 3 disp
recorder Node -file result/duntopdisplace#4-$num.txt -time -node 61 56 -dof 1 2 3 disp
constraints Transformation; 
numberer Plain;  
system UmfPack; 
test NormDispIncr  3 1000; 
algorithm Newton 
integrator Newmark 0.5 0.25 
analysis Transient
analyze 3000 0.02
puts "$num/999"
