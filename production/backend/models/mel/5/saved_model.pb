Љи8
є$Щ$
B
AddV2
x"T
y"T
z"T"
Ttype:
2	
B
AssignVariableOp
resource
value"dtype"
dtypetype
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
P

ComplexAbs
x"T	
y"Tout"
Ttype0:
2"
Touttype0:
2
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype

Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

,
Cos
x"T
y"T"
Ttype:

2
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
?
FloorDiv
x"T
y"T
z"T"
Ttype:
2	
:
FloorMod
x"T
y"T
z"T"
Ttype:
	2	
њ
FusedBatchNormV3
x"T

scale"U
offset"U	
mean"U
variance"U
y"T

batch_mean"U
batch_variance"U
reserve_space_1"U
reserve_space_2"U
reserve_space_3"U"
Ttype:
2"
Utype:
2"
epsilonfloat%Зб8"&
exponential_avg_factorfloat%  ?";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
­
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	

MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
:
Maximum
x"T
y"T
z"T"
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
=
Mul
x"T
y"T
z"T"
Ttype:
2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:

Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
s
RFFT
input"Treal

fft_length
output"Tcomplex"
Trealtype0:
2"
Tcomplextype0:
2
b
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx"
Tidxtype0:

2	
@
ReadVariableOp
resource
value"dtype"
dtypetype
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2

SplitV

value"T
size_splits"Tlen
	split_dim
output"T*	num_split"
	num_splitint(0"	
Ttype"
Tlentype0	:
2	
О
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
@
StaticRegexFullMatch	
input

output
"
patternstring
і
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
;
Sub
x"T
y"T
z"T"
Ttype:
2	
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.4.12v2.4.0-49-g85c8b2a817f8А3

batch_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_namebatch_normalization/gamma

-batch_normalization/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization/gamma*
_output_shapes
:*
dtype0

batch_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_namebatch_normalization/beta

,batch_normalization/beta/Read/ReadVariableOpReadVariableOpbatch_normalization/beta*
_output_shapes
:*
dtype0

batch_normalization/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!batch_normalization/moving_mean

3batch_normalization/moving_mean/Read/ReadVariableOpReadVariableOpbatch_normalization/moving_mean*
_output_shapes
:*
dtype0

#batch_normalization/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#batch_normalization/moving_variance

7batch_normalization/moving_variance/Read/ReadVariableOpReadVariableOp#batch_normalization/moving_variance*
_output_shapes
:*
dtype0
~
conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d/kernel
w
!conv2d/kernel/Read/ReadVariableOpReadVariableOpconv2d/kernel*&
_output_shapes
:*
dtype0
n
conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d/bias
g
conv2d/bias/Read/ReadVariableOpReadVariableOpconv2d/bias*
_output_shapes
:*
dtype0

batch_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_1/gamma

/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_1/gamma*
_output_shapes
:*
dtype0

batch_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namebatch_normalization_1/beta

.batch_normalization_1/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_1/beta*
_output_shapes
:*
dtype0

!batch_normalization_1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!batch_normalization_1/moving_mean

5batch_normalization_1/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_1/moving_mean*
_output_shapes
:*
dtype0
Ђ
%batch_normalization_1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%batch_normalization_1/moving_variance

9batch_normalization_1/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_1/moving_variance*
_output_shapes
:*
dtype0

conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_1/kernel
{
#conv2d_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_1/kernel*&
_output_shapes
:*
dtype0
r
conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_1/bias
k
!conv2d_1/bias/Read/ReadVariableOpReadVariableOpconv2d_1/bias*
_output_shapes
:*
dtype0

batch_normalization_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_2/gamma

/batch_normalization_2/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_2/gamma*
_output_shapes
:*
dtype0

batch_normalization_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namebatch_normalization_2/beta

.batch_normalization_2/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_2/beta*
_output_shapes
:*
dtype0

!batch_normalization_2/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!batch_normalization_2/moving_mean

5batch_normalization_2/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_2/moving_mean*
_output_shapes
:*
dtype0
Ђ
%batch_normalization_2/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%batch_normalization_2/moving_variance

9batch_normalization_2/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_2/moving_variance*
_output_shapes
:*
dtype0

conv2d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nameconv2d_2/kernel
{
#conv2d_2/kernel/Read/ReadVariableOpReadVariableOpconv2d_2/kernel*&
_output_shapes
: *
dtype0
r
conv2d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_2/bias
k
!conv2d_2/bias/Read/ReadVariableOpReadVariableOpconv2d_2/bias*
_output_shapes
: *
dtype0

batch_normalization_3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_3/gamma

/batch_normalization_3/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_3/gamma*
_output_shapes
: *
dtype0

batch_normalization_3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_namebatch_normalization_3/beta

.batch_normalization_3/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_3/beta*
_output_shapes
: *
dtype0

!batch_normalization_3/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!batch_normalization_3/moving_mean

5batch_normalization_3/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_3/moving_mean*
_output_shapes
: *
dtype0
Ђ
%batch_normalization_3/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *6
shared_name'%batch_normalization_3/moving_variance

9batch_normalization_3/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_3/moving_variance*
_output_shapes
: *
dtype0
u
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	D@*
shared_namedense/kernel
n
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes
:	D@*
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:@*
dtype0
x
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*
shared_namedense_1/kernel
q
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes

:@*
dtype0
p
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
:*
dtype0
x
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense_2/kernel
q
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel*
_output_shapes

:*
dtype0
p
dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_2/bias
i
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0

 Adam/batch_normalization/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/batch_normalization/gamma/m

4Adam/batch_normalization/gamma/m/Read/ReadVariableOpReadVariableOp Adam/batch_normalization/gamma/m*
_output_shapes
:*
dtype0

Adam/batch_normalization/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/batch_normalization/beta/m

3Adam/batch_normalization/beta/m/Read/ReadVariableOpReadVariableOpAdam/batch_normalization/beta/m*
_output_shapes
:*
dtype0

Adam/conv2d/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d/kernel/m

(Adam/conv2d/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d/kernel/m*&
_output_shapes
:*
dtype0
|
Adam/conv2d/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/conv2d/bias/m
u
&Adam/conv2d/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d/bias/m*
_output_shapes
:*
dtype0

"Adam/batch_normalization_1/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_1/gamma/m

6Adam/batch_normalization_1/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_1/gamma/m*
_output_shapes
:*
dtype0

!Adam/batch_normalization_1/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/batch_normalization_1/beta/m

5Adam/batch_normalization_1/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_1/beta/m*
_output_shapes
:*
dtype0

Adam/conv2d_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_1/kernel/m

*Adam/conv2d_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/kernel/m*&
_output_shapes
:*
dtype0

Adam/conv2d_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d_1/bias/m
y
(Adam/conv2d_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/bias/m*
_output_shapes
:*
dtype0

"Adam/batch_normalization_2/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_2/gamma/m

6Adam/batch_normalization_2/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_2/gamma/m*
_output_shapes
:*
dtype0

!Adam/batch_normalization_2/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/batch_normalization_2/beta/m

5Adam/batch_normalization_2/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_2/beta/m*
_output_shapes
:*
dtype0

Adam/conv2d_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdam/conv2d_2/kernel/m

*Adam/conv2d_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/kernel/m*&
_output_shapes
: *
dtype0

Adam/conv2d_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/conv2d_2/bias/m
y
(Adam/conv2d_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/bias/m*
_output_shapes
: *
dtype0

"Adam/batch_normalization_3/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"Adam/batch_normalization_3/gamma/m

6Adam/batch_normalization_3/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_3/gamma/m*
_output_shapes
: *
dtype0

!Adam/batch_normalization_3/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!Adam/batch_normalization_3/beta/m

5Adam/batch_normalization_3/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_3/beta/m*
_output_shapes
: *
dtype0

Adam/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	D@*$
shared_nameAdam/dense/kernel/m
|
'Adam/dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/m*
_output_shapes
:	D@*
dtype0
z
Adam/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*"
shared_nameAdam/dense/bias/m
s
%Adam/dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense/bias/m*
_output_shapes
:@*
dtype0

Adam/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*&
shared_nameAdam/dense_1/kernel/m

)Adam/dense_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/m*
_output_shapes

:@*
dtype0
~
Adam/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_1/bias/m
w
'Adam/dense_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/m*
_output_shapes
:*
dtype0

Adam/dense_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameAdam/dense_2/kernel/m

)Adam/dense_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_2/kernel/m*
_output_shapes

:*
dtype0
~
Adam/dense_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_2/bias/m
w
'Adam/dense_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_2/bias/m*
_output_shapes
:*
dtype0

 Adam/batch_normalization/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/batch_normalization/gamma/v

4Adam/batch_normalization/gamma/v/Read/ReadVariableOpReadVariableOp Adam/batch_normalization/gamma/v*
_output_shapes
:*
dtype0

Adam/batch_normalization/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/batch_normalization/beta/v

3Adam/batch_normalization/beta/v/Read/ReadVariableOpReadVariableOpAdam/batch_normalization/beta/v*
_output_shapes
:*
dtype0

Adam/conv2d/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d/kernel/v

(Adam/conv2d/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d/kernel/v*&
_output_shapes
:*
dtype0
|
Adam/conv2d/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/conv2d/bias/v
u
&Adam/conv2d/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d/bias/v*
_output_shapes
:*
dtype0

"Adam/batch_normalization_1/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_1/gamma/v

6Adam/batch_normalization_1/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_1/gamma/v*
_output_shapes
:*
dtype0

!Adam/batch_normalization_1/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/batch_normalization_1/beta/v

5Adam/batch_normalization_1/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_1/beta/v*
_output_shapes
:*
dtype0

Adam/conv2d_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_1/kernel/v

*Adam/conv2d_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/kernel/v*&
_output_shapes
:*
dtype0

Adam/conv2d_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d_1/bias/v
y
(Adam/conv2d_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/bias/v*
_output_shapes
:*
dtype0

"Adam/batch_normalization_2/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_2/gamma/v

6Adam/batch_normalization_2/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_2/gamma/v*
_output_shapes
:*
dtype0

!Adam/batch_normalization_2/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/batch_normalization_2/beta/v

5Adam/batch_normalization_2/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_2/beta/v*
_output_shapes
:*
dtype0

Adam/conv2d_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdam/conv2d_2/kernel/v

*Adam/conv2d_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/kernel/v*&
_output_shapes
: *
dtype0

Adam/conv2d_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/conv2d_2/bias/v
y
(Adam/conv2d_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/bias/v*
_output_shapes
: *
dtype0

"Adam/batch_normalization_3/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"Adam/batch_normalization_3/gamma/v

6Adam/batch_normalization_3/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_3/gamma/v*
_output_shapes
: *
dtype0

!Adam/batch_normalization_3/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!Adam/batch_normalization_3/beta/v

5Adam/batch_normalization_3/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_3/beta/v*
_output_shapes
: *
dtype0

Adam/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	D@*$
shared_nameAdam/dense/kernel/v
|
'Adam/dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/v*
_output_shapes
:	D@*
dtype0
z
Adam/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*"
shared_nameAdam/dense/bias/v
s
%Adam/dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense/bias/v*
_output_shapes
:@*
dtype0

Adam/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*&
shared_nameAdam/dense_1/kernel/v

)Adam/dense_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/v*
_output_shapes

:@*
dtype0
~
Adam/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_1/bias/v
w
'Adam/dense_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/v*
_output_shapes
:*
dtype0

Adam/dense_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameAdam/dense_2/kernel/v

)Adam/dense_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_2/kernel/v*
_output_shapes

:*
dtype0
~
Adam/dense_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_2/bias/v
w
'Adam/dense_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_2/bias/v*
_output_shapes
:*
dtype0
т
ConstConst* 
_output_shapes
:
*
dtype0*Ё
valueB
"                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                ѓъ<                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            xш<рПk<                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            Z.=ЅіQ9                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        +0e<рПы<                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            дц<Лo<                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            -=Ѕіб9                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        Pшa<Юcэ<                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            +0х<Or<                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            Ж,=ћx:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        v ^<Ля<                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            =у<pu<                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             ф+=ЅіQ:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        X[<ЇЋ№<                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            Pшс<Jпx<                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            )+=&::                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        РX<Oђ<                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            bDр<%'|<                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            3@*=ћx:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        цШT<ѓѓ<                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            v о<џn<                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            <n)=ЯЗЗ:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        Q<pѕ<                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            ќм<m[<                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            F(=Ѕіб:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        19N<];ї<                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            Xл<[џ<                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            OЪ'=y5ь:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        VёJ<Jпј<                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            ЎДй<GЃ<                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            Xј&=&:;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        {ЉG<8њ<                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            Ри<5G<                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            b&&=Y;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        ЂaD<%'ќ<                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            гlж<"ы<                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            kT%=ћx;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        ЧA<Ы§<                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            цШд<<                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            t$=e*;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        ьб=<џnџ<                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            ј$г<§2<                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            }А#=ЯЗ7;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        :<v =                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            б<ъж<                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            о"=;зD;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        8B7<m[=                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            нЯ<зz<                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            "=ЅіQ;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        ]њ3<c-=                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            19Ю<Х<                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            :!=_;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        В0<[џ=                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            DЬ<ВТ<                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            Ѓh =y5l;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        Їj-<Qб=                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            VёЪ<f<                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            Ќ=уTy;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        Э"*<GЃ=                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            iMЩ<
<                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            ЖФ=&:;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        ђк&<>u=                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            {ЉЧ<+<                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            бt=н;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        ї('<{=                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            аЪ<<                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            Еъ=_й:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        yн]<Юе<                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            Tш<Fв.<                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        вN;=                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            zЃ<K#<                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            =n(;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        ЬW<JТ<                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            nGщ<F<                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        qЇ <и{х<                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            OР<ЅB<                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        eЈ|;|ў<                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            -ўЁ<DФp<                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        Ф7:=                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            р~<Л<                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            Ф9=y):                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        XH~<~J<                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            ЈЮ§<АЕ:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        щr<Mш<                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            яї<%Ў:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        §оr<Z]<                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            ј<пУ:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        <nФt<                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        ЅН_:P№<                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            пЌ<uAS<                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
t>;yТй<                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            9<јi)<                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        дГ;(П<                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            ­<Й№;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        0
< Ђ<                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            ~KФ<б?;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        ЬA<х<                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            _Эн<є9                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        y,~<t=<                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        g|;)=Л<                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            <n<фјч;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        }Y
<6<                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            cС<ір;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        W<E"O<                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        ~1,;'Л<                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            о<эя;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        сC<rќ<                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            ыЛ<Іfњ:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        БпZ<FЦ7<                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        7Жz;мЈ<                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            Э(<Љћ­;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        (z<pIh<                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        Z+*:ЪМ<                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             "< <                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        6нф;ѓ<                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            NА<ѕщЫ:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        7tY<Вl<                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        ЯЄ;Ў<                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            $Њ<YЌD;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        ><Dф.<                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        Lћv;ЙN<                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            ѓ<Oxy;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        E.<м+6<                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        в,N;вќ<                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            Њ:<Щn;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        H'<!a4<                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        ч<J;nE<                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            Q<ыw;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        Б0)<i*<                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         §f;Њ<                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            D<JЫG;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        W2<<<                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        чI;дЃ<                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            ш <Иf§:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        ВB<Лu<                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        'ЛЙ;>e<                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            бњ<њії9                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        FW<гУЫ;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        в1ю;D<                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        ^TЕ:&<                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            чћq<э;;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        м<Q<                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        IЯh;x<                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            с]<еВњ:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        Q+9<jЁщ;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        П5У;KJ<                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        юІ :Ет<                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            Г_<:;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        B<<                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        &Dc;k<                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            I<I"Ц:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        T9<ХЬ;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        .,д;Ю 4<                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        бд:Я<                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            i<dњD;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        ЕX<jі;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        О'Ѓ;§+E<                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        сЙ9W<                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            О#R<й';                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        .№	<<                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        ;y;H
N<                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            u<nР9                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        Ё?C<1Z;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        ­фќ;МV<                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        2f;`O<                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            $}<&:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        ;<Т;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        .є;a
<                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        fаa;6aJ<                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            тЮx<x:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        %U:<Љы;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        аЖї;r<                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        Ќu;/o?<                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            Oz<9C8                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        уЦ><fl;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        1><Yъ;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        ўj;?]/<                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        йЬB:гzi<                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            P<H<.70;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        5<ТнЦ;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        3­;іЯ<                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        ђW№:1R<                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            ]V<Ц:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         <Щ1;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        яг;w^<                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        ЫЁO;
$7<                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            іТg<| 9                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        C?4<Ё'Q;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        Л <Ю&Э;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        Кo;хм<                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        NЁЭ:d&K<                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            YЅK<MР:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        <л;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        яв;FВя;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        a;Ф'<                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        њGщ9нЏW<                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            ёј6<;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        м1<Љ;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        еВ;M5<                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        Т*;п/<                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            ,U<]v9                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        ыљ(<Ws=;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        UЭј;jРЕ;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        гІ;c<                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        Ѓ ;єц1<                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            gmK<Ш
:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        oѓ <H;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        эђь;4З;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        ўў;,<                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        ;.<                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            ИF<WG:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        Є><>^?;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        Pы;SЕЎ;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        Y;Л§;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        ФB;н`&<                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            йгF<QјY9                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        ПA <ц:$;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        I_ѓ;$k;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        ;І;еИш;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        Р-2;C<                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        Е*П9Њ?<                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            $o&<Bђ:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        ђ­<зf;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        йЙ;н(Ь;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        <Ў`;rѕ	<                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        ыR:uж-<                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            D0<m:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        e><iFI;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        rpд;MЉ;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        d;feэ;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        Џ;?у<                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        й8Ь;<                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            \ъ<ђќ:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        Аѕ;a/;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        ЊTВ;FXС;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        E)_;@<                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        oRГ:е!<                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            E-<}#:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        Hw<і$;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        Sл;Э;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        ЃЗ;KЊЮ;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        \88;х`<                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        Тd:Ѕl%<                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            F"<O:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        уM<OЅ7;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        џўЫ;Л§;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        9b;Ю(в;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        ц%;ёЉ<                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        eE1:{?$<                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            №Н<j:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        БЙџ;N8;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        їХ;х;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        Q5;ЫъЬ;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        Aц$;XІ<                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        E:Kз<                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             #<еv:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        A<џ;љ(;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        A2Ш;7;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        A(;^ёП;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        <4;ЊЋѕ;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        Q:ћВ<                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            O<+:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        @<Ў;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        ъЙб;эq;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        SG;:)Ќ;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        xЉQ;ъ[п;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        б:NG	<                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            Ъа"<7Фє5                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        г	<cУ:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        ОЌс;ыDC;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        TВЏ;l;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        бo{; 5У;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        ќz;5џѓ;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        N:dd<                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            Ѓm<Ђ54:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        E;ї;v
;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        BЧ;g;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        @ћ;HџЁ;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |ЖP;Я|а;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        юьт:Uњў;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        Г9юЛ<                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            г<R:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        Eф;C ;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        уЖ;тнx;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        &;МЈ;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        ]>8;A	е;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        рєК:5Ћ <                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         -8Щб<                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            B<аЃ:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        Уй;ЈV&;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        Ў;Хz;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        ЋE;­Ї;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        q0;каб;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        Г:ќ;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        ЪiD8<                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            Pџџ;Џ3:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         Щж;;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        ё­;,n;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        A^;ы{;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        "Q6;@ЖЧ;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        ЫЧ:№я;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        г9v<                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            кр <Ѓq:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        М|к;Њ	;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        Ф7Г;ОU;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        Ьђ;М4;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        Ј[I;/З;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        qЃѕ:Ђпн;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        $1:<                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            ЉП<д№ў9                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        Аф;Ъйб:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        ЈО;Аћ1;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        h<;{
{;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        Ёg;ЃЂ;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        EЪ;Ц;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        ќхЃ:nы;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        њvу8iб<                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            oеђ;у|:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        з,Я;Ћ;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        @Ћ;н=J;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        Јл;ю;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         fH;!НЊ;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        №;9Э;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        g:S[№;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            џ<ћДI9                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        5ф;Пф:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        m"Т;pI;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        Є' ; S;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        ЕY|;Шћ;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        $d8;P'Ќ;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        #нш:зRЭ;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        ўуA:_~ю;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            3оќ;u9                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        ќ|м;НЏЂ:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        ХМ;N;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        К;ОЦO;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        ЏВv;;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        A№5;ЯЇ;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        І[ъ:ЖЦ;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        ­Q:>Rц;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            яњ;'Љ79                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        ,л;p:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        QМ;~ѕ;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        &v;є2@;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        p6};kp|;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        ?;№V;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        ЛЪ;ЋuК;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        Р):fи;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        ІрЫ8!Гі;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            Nqп;ЊшM:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        I
Т;Уй:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        FЃЄ;фH&;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        A<;@А_;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        yЊS;Ю;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        qм;}?Љ;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        ЯМ:*ѓХ;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |:йІт;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            ЅЂш;№FЏ9                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        Ь;Ђ8:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        iА;ФO;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        Ы;7:;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        [ q;ЊЖp;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        9;Е;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        т;ШЏ;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        M	:hЪ;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        Sл9;Тх;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            pл;bP:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        ДНР;lшА:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        Э
І;S;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        цW;rД@;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        ћIa;дt;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        -ф+;Wz;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        Мќь:eЎ;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        1:tШ;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        ,;9Њт;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            ўд;I(:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        цЛ;ЫыА:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        ЯЁ;Й!
;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        З.;Э;;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        @{];_ym;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        *;;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        Фmя:hЈ;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        gЉ:m>С;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        %9Vк;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            %г;ыѕ9                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        чК;З№:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        ЊЈЂ;ѓњ:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        <j;Ђ,;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        Wd;Жї[;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        Ок3;jІ;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        с];љP;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        	ТЅ:ћД;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        	:ІЬ;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            фЗж;гЭQ9                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        ­П;кнh:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        xЈ; ЄЮ:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        Ci;Љl;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        t;CA;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        ЏiF;мЁn;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        C5;;о;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        Џд:kЄ;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        Б1o:дјК;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        Y9!б;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             Ш;gЎђ9                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        BВ;|Ё:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        ;`ш:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        У;ЁF;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         a;AJ;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        5;<u;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        	;Л;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        К:3Ѕ;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        9D:ЌЛ;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        љQ8$а;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            gР;M:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        КlЋ;O:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        Tr;zь:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        яw;ь1;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        ћX;Ю&H;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        J/;Бq;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        ;I;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        j9Ж:КЁ;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        ЌD:+§Е;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        dц8wЪ;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            лkН;оR:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        RnЉ;оЗ:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        Шp;LЦм:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        ?s;^j;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        jыZ;q<;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        V№2;Ьxc;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        Cѕ
;@;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        bєХ:У;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        wќk:9GЌ;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        U 9еЪП;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            ВП;и9                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        	Ќ;Хb:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        ќ;#ФЛ:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        ёя;м;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        ЗЦe;ЈC(;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        ­?;stM;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        `;=Ѕr;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        jіц:ы;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        Ф:j;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        x#:ЯБ;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        aй75ДУ;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            *ВВ;X	:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        / ;:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        5d;nв:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        uzx;Ї;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        ,T;0;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        о/;S;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        ;"љv;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        IЮ:г4;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        aщ:э;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        т5ѕ9XЅА;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            ГщМ;GЈ8                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        Ћ;K:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        VQ;й:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        (;$е:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        єqo;ЁW;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        йL;ћ.;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        ;A*;TтO;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        оЈ;ЏЇq;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        !Ъ:Ж;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        L№:2;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        Nўў9о{Ћ;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            ЋОЙ;{§!8                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        ќBЉ;й
:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        MЧ;Щ:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        K;Д%Ц:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        мo;&A;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |ЈN;ro#;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        Б-;ОC;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        ПЙ;	Ьc;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        Пз:+§;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        :Q;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        N':w+Ђ;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        8Ф9BВ;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            sЋ;cГ9                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        qd;d[T:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        oЏ;Ї:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        лєw;Tзф:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        иX;;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        д 9;zР/;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        аЖ;уjN;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        є:Km;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        ХЕ:кп;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        уm:5;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        vр9BЄ;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            ЌЂА;вG8                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        §ЊЁ;H`:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        NГ;hCv:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        Л;DЕ:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        рi;д№:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        K;2;;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        %Љ-;ћs2;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        ШЙ;УЌO;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        ду:хl;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        ЖЇ:);                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        ИЎW:Ћ;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        тП9ёGЂ;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            H'Ћ;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            ф;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            М ;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            v];                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            _4d;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            г­G;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            H'+;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            М ;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            _4ф:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            H'Ћ:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            _4d:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            _4ф9                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                

NoOpNoOp

Const_1Const"/device:CPU:0*
_output_shapes
: *
dtype0*П
valueДBА BЈ
ч
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
layer-7
	layer_with_weights-4
	layer-8

layer_with_weights-5

layer-9
layer-10
layer_with_weights-6
layer-11
layer-12
layer_with_weights-7
layer-13
layer-14
layer_with_weights-8
layer-15
layer-16
layer_with_weights-9
layer-17
	optimizer
regularization_losses
	variables
trainable_variables
	keras_api

signatures
R
regularization_losses
	variables
trainable_variables
	keras_api
y
layer-0
layer-1
layer-2
 regularization_losses
!	variables
"trainable_variables
#	keras_api

$axis
	%gamma
&beta
'moving_mean
(moving_variance
)regularization_losses
*	variables
+trainable_variables
,	keras_api
h

-kernel
.bias
/regularization_losses
0	variables
1trainable_variables
2	keras_api
R
3regularization_losses
4	variables
5trainable_variables
6	keras_api

7axis
	8gamma
9beta
:moving_mean
;moving_variance
<regularization_losses
=	variables
>trainable_variables
?	keras_api
h

@kernel
Abias
Bregularization_losses
C	variables
Dtrainable_variables
E	keras_api
R
Fregularization_losses
G	variables
Htrainable_variables
I	keras_api

Jaxis
	Kgamma
Lbeta
Mmoving_mean
Nmoving_variance
Oregularization_losses
P	variables
Qtrainable_variables
R	keras_api
h

Skernel
Tbias
Uregularization_losses
V	variables
Wtrainable_variables
X	keras_api
R
Yregularization_losses
Z	variables
[trainable_variables
\	keras_api

]axis
	^gamma
_beta
`moving_mean
amoving_variance
bregularization_losses
c	variables
dtrainable_variables
e	keras_api
R
fregularization_losses
g	variables
htrainable_variables
i	keras_api
h

jkernel
kbias
lregularization_losses
m	variables
ntrainable_variables
o	keras_api
R
pregularization_losses
q	variables
rtrainable_variables
s	keras_api
h

tkernel
ubias
vregularization_losses
w	variables
xtrainable_variables
y	keras_api
R
zregularization_losses
{	variables
|trainable_variables
}	keras_api
l

~kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
е
	iter
beta_1
beta_2

decay
learning_rate%m&m-m.m8m9m@mAmKmLmSmTm^m_mjmkmtmum ~mЁmЂ%vЃ&vЄ-vЅ.vІ8vЇ9vЈ@vЉAvЊKvЋLvЌSv­TvЎ^vЏ_vАjvБkvВtvГuvД~vЕvЖ
 
ж
%0
&1
'2
(3
-4
.5
86
97
:8
;9
@10
A11
K12
L13
M14
N15
S16
T17
^18
_19
`20
a21
j22
k23
t24
u25
~26
27

%0
&1
-2
.3
84
95
@6
A7
K8
L9
S10
T11
^12
_13
j14
k15
t16
u17
~18
19
В
regularization_losses
 layer_regularization_losses
layers
metrics
non_trainable_variables
	variables
layer_metrics
trainable_variables
 
 
 
 
В
regularization_losses
 layer_regularization_losses
layers
metrics
non_trainable_variables
	variables
layer_metrics
trainable_variables
V
regularization_losses
	variables
trainable_variables
	keras_api
V
regularization_losses
	variables
trainable_variables
	keras_api
n
filterbank_kwargs
regularization_losses
	variables
trainable_variables
	keras_api
 
 
 
В
 regularization_losses
  layer_regularization_losses
Ёlayers
Ђmetrics
Ѓnon_trainable_variables
!	variables
Єlayer_metrics
"trainable_variables
 
db
VARIABLE_VALUEbatch_normalization/gamma5layer_with_weights-0/gamma/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEbatch_normalization/beta4layer_with_weights-0/beta/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEbatch_normalization/moving_mean;layer_with_weights-0/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE#batch_normalization/moving_variance?layer_with_weights-0/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

%0
&1
'2
(3

%0
&1
В
)regularization_losses
 Ѕlayer_regularization_losses
Іlayers
Їmetrics
Јnon_trainable_variables
*	variables
Љlayer_metrics
+trainable_variables
YW
VARIABLE_VALUEconv2d/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv2d/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

-0
.1

-0
.1
В
/regularization_losses
 Њlayer_regularization_losses
Ћlayers
Ќmetrics
­non_trainable_variables
0	variables
Ўlayer_metrics
1trainable_variables
 
 
 
В
3regularization_losses
 Џlayer_regularization_losses
Аlayers
Бmetrics
Вnon_trainable_variables
4	variables
Гlayer_metrics
5trainable_variables
 
fd
VARIABLE_VALUEbatch_normalization_1/gamma5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_1/beta4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_1/moving_mean;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_1/moving_variance?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

80
91
:2
;3

80
91
В
<regularization_losses
 Дlayer_regularization_losses
Еlayers
Жmetrics
Зnon_trainable_variables
=	variables
Иlayer_metrics
>trainable_variables
[Y
VARIABLE_VALUEconv2d_1/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_1/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

@0
A1

@0
A1
В
Bregularization_losses
 Йlayer_regularization_losses
Кlayers
Лmetrics
Мnon_trainable_variables
C	variables
Нlayer_metrics
Dtrainable_variables
 
 
 
В
Fregularization_losses
 Оlayer_regularization_losses
Пlayers
Рmetrics
Сnon_trainable_variables
G	variables
Тlayer_metrics
Htrainable_variables
 
fd
VARIABLE_VALUEbatch_normalization_2/gamma5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_2/beta4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_2/moving_mean;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_2/moving_variance?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

K0
L1
M2
N3

K0
L1
В
Oregularization_losses
 Уlayer_regularization_losses
Фlayers
Хmetrics
Цnon_trainable_variables
P	variables
Чlayer_metrics
Qtrainable_variables
[Y
VARIABLE_VALUEconv2d_2/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_2/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE
 

S0
T1

S0
T1
В
Uregularization_losses
 Шlayer_regularization_losses
Щlayers
Ъmetrics
Ыnon_trainable_variables
V	variables
Ьlayer_metrics
Wtrainable_variables
 
 
 
В
Yregularization_losses
 Эlayer_regularization_losses
Юlayers
Яmetrics
аnon_trainable_variables
Z	variables
бlayer_metrics
[trainable_variables
 
fd
VARIABLE_VALUEbatch_normalization_3/gamma5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_3/beta4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_3/moving_mean;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_3/moving_variance?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

^0
_1
`2
a3

^0
_1
В
bregularization_losses
 вlayer_regularization_losses
гlayers
дmetrics
еnon_trainable_variables
c	variables
жlayer_metrics
dtrainable_variables
 
 
 
В
fregularization_losses
 зlayer_regularization_losses
иlayers
йmetrics
кnon_trainable_variables
g	variables
лlayer_metrics
htrainable_variables
XV
VARIABLE_VALUEdense/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
dense/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE
 

j0
k1

j0
k1
В
lregularization_losses
 мlayer_regularization_losses
нlayers
оmetrics
пnon_trainable_variables
m	variables
рlayer_metrics
ntrainable_variables
 
 
 
В
pregularization_losses
 сlayer_regularization_losses
тlayers
уmetrics
фnon_trainable_variables
q	variables
хlayer_metrics
rtrainable_variables
ZX
VARIABLE_VALUEdense_1/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_1/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE
 

t0
u1

t0
u1
В
vregularization_losses
 цlayer_regularization_losses
чlayers
шmetrics
щnon_trainable_variables
w	variables
ъlayer_metrics
xtrainable_variables
 
 
 
В
zregularization_losses
 ыlayer_regularization_losses
ьlayers
эmetrics
юnon_trainable_variables
{	variables
яlayer_metrics
|trainable_variables
ZX
VARIABLE_VALUEdense_2/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_2/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE
 

~0
1

~0
1
Е
regularization_losses
 №layer_regularization_losses
ёlayers
ђmetrics
ѓnon_trainable_variables
	variables
єlayer_metrics
trainable_variables
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
 

0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17

ѕ0
і1
8
'0
(1
:2
;3
M4
N5
`6
a7
 
 
 
 
 
 
 
 
 
Е
regularization_losses
 їlayer_regularization_losses
јlayers
љmetrics
њnon_trainable_variables
	variables
ћlayer_metrics
trainable_variables
 
 
 
Е
regularization_losses
 ќlayer_regularization_losses
§layers
ўmetrics
џnon_trainable_variables
	variables
layer_metrics
trainable_variables
 
 
 
 
Е
regularization_losses
 layer_regularization_losses
layers
metrics
non_trainable_variables
	variables
layer_metrics
trainable_variables
 

0
1
2
 
 
 
 
 
 

'0
(1
 
 
 
 
 
 
 
 
 
 
 
 
 
 

:0
;1
 
 
 
 
 
 
 
 
 
 
 
 
 
 

M0
N1
 
 
 
 
 
 
 
 
 
 
 
 
 
 

`0
a1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
8

total

count
	variables
	keras_api
I

total

count

_fn_kwargs
	variables
	keras_api
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

0
1

	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

	variables

VARIABLE_VALUE Adam/batch_normalization/gamma/mQlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/batch_normalization/beta/mPlayer_with_weights-0/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv2d/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adam/batch_normalization_1/gamma/mQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE!Adam/batch_normalization_1/beta/mPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_1/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_1/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adam/batch_normalization_2/gamma/mQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE!Adam/batch_normalization_2/beta/mPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_2/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_2/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adam/batch_normalization_3/gamma/mQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE!Adam/batch_normalization_3/beta/mPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense/kernel/mRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense/bias/mPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_1/kernel/mRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_1/bias/mPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_2/kernel/mRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_2/bias/mPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE Adam/batch_normalization/gamma/vQlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/batch_normalization/beta/vPlayer_with_weights-0/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv2d/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adam/batch_normalization_1/gamma/vQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE!Adam/batch_normalization_1/beta/vPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_1/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_1/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adam/batch_normalization_2/gamma/vQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE!Adam/batch_normalization_2/beta/vPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_2/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_2/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adam/batch_normalization_3/gamma/vQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE!Adam/batch_normalization_3/beta/vPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense/kernel/vRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense/bias/vPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_1/kernel/vRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_1/bias/vPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_2/kernel/vRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_2/bias/vPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

serving_default_reshape_inputPlaceholder*)
_output_shapes
:џџџџџџџџџт	*
dtype0*
shape:џџџџџџџџџт	
­
StatefulPartitionedCallStatefulPartitionedCallserving_default_reshape_inputConstbatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_varianceconv2d/kernelconv2d/biasbatch_normalization_1/gammabatch_normalization_1/beta!batch_normalization_1/moving_mean%batch_normalization_1/moving_varianceconv2d_1/kernelconv2d_1/biasbatch_normalization_2/gammabatch_normalization_2/beta!batch_normalization_2/moving_mean%batch_normalization_2/moving_varianceconv2d_2/kernelconv2d_2/biasbatch_normalization_3/gammabatch_normalization_3/beta!batch_normalization_3/moving_mean%batch_normalization_3/moving_variancedense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/bias*)
Tin"
 2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*>
_read_only_resource_inputs 
	
*-
config_proto

CPU

GPU 2J 8 *,
f'R%
#__inference_signature_wrapper_21447
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Ф
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename-batch_normalization/gamma/Read/ReadVariableOp,batch_normalization/beta/Read/ReadVariableOp3batch_normalization/moving_mean/Read/ReadVariableOp7batch_normalization/moving_variance/Read/ReadVariableOp!conv2d/kernel/Read/ReadVariableOpconv2d/bias/Read/ReadVariableOp/batch_normalization_1/gamma/Read/ReadVariableOp.batch_normalization_1/beta/Read/ReadVariableOp5batch_normalization_1/moving_mean/Read/ReadVariableOp9batch_normalization_1/moving_variance/Read/ReadVariableOp#conv2d_1/kernel/Read/ReadVariableOp!conv2d_1/bias/Read/ReadVariableOp/batch_normalization_2/gamma/Read/ReadVariableOp.batch_normalization_2/beta/Read/ReadVariableOp5batch_normalization_2/moving_mean/Read/ReadVariableOp9batch_normalization_2/moving_variance/Read/ReadVariableOp#conv2d_2/kernel/Read/ReadVariableOp!conv2d_2/bias/Read/ReadVariableOp/batch_normalization_3/gamma/Read/ReadVariableOp.batch_normalization_3/beta/Read/ReadVariableOp5batch_normalization_3/moving_mean/Read/ReadVariableOp9batch_normalization_3/moving_variance/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp"dense_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp4Adam/batch_normalization/gamma/m/Read/ReadVariableOp3Adam/batch_normalization/beta/m/Read/ReadVariableOp(Adam/conv2d/kernel/m/Read/ReadVariableOp&Adam/conv2d/bias/m/Read/ReadVariableOp6Adam/batch_normalization_1/gamma/m/Read/ReadVariableOp5Adam/batch_normalization_1/beta/m/Read/ReadVariableOp*Adam/conv2d_1/kernel/m/Read/ReadVariableOp(Adam/conv2d_1/bias/m/Read/ReadVariableOp6Adam/batch_normalization_2/gamma/m/Read/ReadVariableOp5Adam/batch_normalization_2/beta/m/Read/ReadVariableOp*Adam/conv2d_2/kernel/m/Read/ReadVariableOp(Adam/conv2d_2/bias/m/Read/ReadVariableOp6Adam/batch_normalization_3/gamma/m/Read/ReadVariableOp5Adam/batch_normalization_3/beta/m/Read/ReadVariableOp'Adam/dense/kernel/m/Read/ReadVariableOp%Adam/dense/bias/m/Read/ReadVariableOp)Adam/dense_1/kernel/m/Read/ReadVariableOp'Adam/dense_1/bias/m/Read/ReadVariableOp)Adam/dense_2/kernel/m/Read/ReadVariableOp'Adam/dense_2/bias/m/Read/ReadVariableOp4Adam/batch_normalization/gamma/v/Read/ReadVariableOp3Adam/batch_normalization/beta/v/Read/ReadVariableOp(Adam/conv2d/kernel/v/Read/ReadVariableOp&Adam/conv2d/bias/v/Read/ReadVariableOp6Adam/batch_normalization_1/gamma/v/Read/ReadVariableOp5Adam/batch_normalization_1/beta/v/Read/ReadVariableOp*Adam/conv2d_1/kernel/v/Read/ReadVariableOp(Adam/conv2d_1/bias/v/Read/ReadVariableOp6Adam/batch_normalization_2/gamma/v/Read/ReadVariableOp5Adam/batch_normalization_2/beta/v/Read/ReadVariableOp*Adam/conv2d_2/kernel/v/Read/ReadVariableOp(Adam/conv2d_2/bias/v/Read/ReadVariableOp6Adam/batch_normalization_3/gamma/v/Read/ReadVariableOp5Adam/batch_normalization_3/beta/v/Read/ReadVariableOp'Adam/dense/kernel/v/Read/ReadVariableOp%Adam/dense/bias/v/Read/ReadVariableOp)Adam/dense_1/kernel/v/Read/ReadVariableOp'Adam/dense_1/bias/v/Read/ReadVariableOp)Adam/dense_2/kernel/v/Read/ReadVariableOp'Adam/dense_2/bias/v/Read/ReadVariableOpConst_1*Z
TinS
Q2O	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *'
f"R 
__inference__traced_save_23502
Й
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamebatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_varianceconv2d/kernelconv2d/biasbatch_normalization_1/gammabatch_normalization_1/beta!batch_normalization_1/moving_mean%batch_normalization_1/moving_varianceconv2d_1/kernelconv2d_1/biasbatch_normalization_2/gammabatch_normalization_2/beta!batch_normalization_2/moving_mean%batch_normalization_2/moving_varianceconv2d_2/kernelconv2d_2/biasbatch_normalization_3/gammabatch_normalization_3/beta!batch_normalization_3/moving_mean%batch_normalization_3/moving_variancedense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1 Adam/batch_normalization/gamma/mAdam/batch_normalization/beta/mAdam/conv2d/kernel/mAdam/conv2d/bias/m"Adam/batch_normalization_1/gamma/m!Adam/batch_normalization_1/beta/mAdam/conv2d_1/kernel/mAdam/conv2d_1/bias/m"Adam/batch_normalization_2/gamma/m!Adam/batch_normalization_2/beta/mAdam/conv2d_2/kernel/mAdam/conv2d_2/bias/m"Adam/batch_normalization_3/gamma/m!Adam/batch_normalization_3/beta/mAdam/dense/kernel/mAdam/dense/bias/mAdam/dense_1/kernel/mAdam/dense_1/bias/mAdam/dense_2/kernel/mAdam/dense_2/bias/m Adam/batch_normalization/gamma/vAdam/batch_normalization/beta/vAdam/conv2d/kernel/vAdam/conv2d/bias/v"Adam/batch_normalization_1/gamma/v!Adam/batch_normalization_1/beta/vAdam/conv2d_1/kernel/vAdam/conv2d_1/bias/v"Adam/batch_normalization_2/gamma/v!Adam/batch_normalization_2/beta/vAdam/conv2d_2/kernel/vAdam/conv2d_2/bias/v"Adam/batch_normalization_3/gamma/v!Adam/batch_normalization_3/beta/vAdam/dense/kernel/vAdam/dense/bias/vAdam/dense_1/kernel/vAdam/dense_1/bias/vAdam/dense_2/kernel/vAdam/dense_2/bias/v*Y
TinR
P2N*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference__traced_restore_23743шЂ 
г
Ј
5__inference_batch_normalization_2_layer_call_fn_22802

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityЂStatefulPartitionedCall 
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ" *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_206962
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:џџџџџџџџџ" 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ" ::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ" 
 
_user_specified_nameinputs
њ
э
E__inference_sequential_layer_call_and_return_conditional_losses_21965

inputs/
+melspectrogram_apply_filterbank_tensordot_b/
+batch_normalization_readvariableop_resource1
-batch_normalization_readvariableop_1_resource@
<batch_normalization_fusedbatchnormv3_readvariableop_resourceB
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource1
-batch_normalization_1_readvariableop_resource3
/batch_normalization_1_readvariableop_1_resourceB
>batch_normalization_1_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource1
-batch_normalization_2_readvariableop_resource3
/batch_normalization_2_readvariableop_1_resourceB
>batch_normalization_2_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource+
'conv2d_2_conv2d_readvariableop_resource,
(conv2d_2_biasadd_readvariableop_resource1
-batch_normalization_3_readvariableop_resource3
/batch_normalization_3_readvariableop_1_resourceB
>batch_normalization_3_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource
identityЂ3batch_normalization/FusedBatchNormV3/ReadVariableOpЂ5batch_normalization/FusedBatchNormV3/ReadVariableOp_1Ђ"batch_normalization/ReadVariableOpЂ$batch_normalization/ReadVariableOp_1Ђ5batch_normalization_1/FusedBatchNormV3/ReadVariableOpЂ7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1Ђ$batch_normalization_1/ReadVariableOpЂ&batch_normalization_1/ReadVariableOp_1Ђ5batch_normalization_2/FusedBatchNormV3/ReadVariableOpЂ7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1Ђ$batch_normalization_2/ReadVariableOpЂ&batch_normalization_2/ReadVariableOp_1Ђ5batch_normalization_3/FusedBatchNormV3/ReadVariableOpЂ7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1Ђ$batch_normalization_3/ReadVariableOpЂ&batch_normalization_3/ReadVariableOp_1Ђconv2d/BiasAdd/ReadVariableOpЂconv2d/Conv2D/ReadVariableOpЂconv2d_1/BiasAdd/ReadVariableOpЂconv2d_1/Conv2D/ReadVariableOpЂconv2d_2/BiasAdd/ReadVariableOpЂconv2d_2/Conv2D/ReadVariableOpЂdense/BiasAdd/ReadVariableOpЂdense/MatMul/ReadVariableOpЂdense_1/BiasAdd/ReadVariableOpЂdense_1/MatMul/ReadVariableOpЂdense_2/BiasAdd/ReadVariableOpЂdense_2/MatMul/ReadVariableOpT
reshape/ShapeShapeinputs*
T0*
_output_shapes
:2
reshape/Shape
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape/strided_slice/stack
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_1
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_2
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape/strided_slicev
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB	 :т	2
reshape/Reshape/shape/1t
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/2Ш
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape/Reshape/shape
reshape/ReshapeReshapeinputsreshape/Reshape/shape:output:0*
T0*-
_output_shapes
:џџџџџџџџџт	2
reshape/Reshape
"melspectrogram/stft/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2$
"melspectrogram/stft/transpose/permЪ
melspectrogram/stft/transpose	Transposereshape/Reshape:output:0+melspectrogram/stft/transpose/perm:output:0*
T0*-
_output_shapes
:џџџџџџџџџт	2
melspectrogram/stft/transposeЏ
4melspectrogram/stft/stft_tf.signal.stft/frame_lengthConst*
_output_shapes
: *
dtype0*
value
B :26
4melspectrogram/stft/stft_tf.signal.stft/frame_lengthЋ
2melspectrogram/stft/stft_tf.signal.stft/frame_stepConst*
_output_shapes
: *
dtype0*
value
B :24
2melspectrogram/stft/stft_tf.signal.stft/frame_stepЋ
2melspectrogram/stft/stft_tf.signal.stft/fft_lengthConst*
_output_shapes
: *
dtype0*
value
B :24
2melspectrogram/stft/stft_tf.signal.stft/fft_lengthГ
2melspectrogram/stft/stft_tf.signal.stft/frame/axisConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ24
2melspectrogram/stft/stft_tf.signal.stft/frame/axisЛ
3melspectrogram/stft/stft_tf.signal.stft/frame/ShapeShape!melspectrogram/stft/transpose:y:0*
T0*
_output_shapes
:25
3melspectrogram/stft/stft_tf.signal.stft/frame/ShapeЊ
2melspectrogram/stft/stft_tf.signal.stft/frame/RankConst*
_output_shapes
: *
dtype0*
value	B :24
2melspectrogram/stft/stft_tf.signal.stft/frame/RankИ
9melspectrogram/stft/stft_tf.signal.stft/frame/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2;
9melspectrogram/stft/stft_tf.signal.stft/frame/range/startИ
9melspectrogram/stft/stft_tf.signal.stft/frame/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2;
9melspectrogram/stft/stft_tf.signal.stft/frame/range/deltaд
3melspectrogram/stft/stft_tf.signal.stft/frame/rangeRangeBmelspectrogram/stft/stft_tf.signal.stft/frame/range/start:output:0;melspectrogram/stft/stft_tf.signal.stft/frame/Rank:output:0Bmelspectrogram/stft/stft_tf.signal.stft/frame/range/delta:output:0*
_output_shapes
:25
3melspectrogram/stft/stft_tf.signal.stft/frame/rangeй
Amelspectrogram/stft/stft_tf.signal.stft/frame/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2C
Amelspectrogram/stft/stft_tf.signal.stft/frame/strided_slice/stackд
Cmelspectrogram/stft/stft_tf.signal.stft/frame/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2E
Cmelspectrogram/stft/stft_tf.signal.stft/frame/strided_slice/stack_1д
Cmelspectrogram/stft/stft_tf.signal.stft/frame/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2E
Cmelspectrogram/stft/stft_tf.signal.stft/frame/strided_slice/stack_2і
;melspectrogram/stft/stft_tf.signal.stft/frame/strided_sliceStridedSlice<melspectrogram/stft/stft_tf.signal.stft/frame/range:output:0Jmelspectrogram/stft/stft_tf.signal.stft/frame/strided_slice/stack:output:0Lmelspectrogram/stft/stft_tf.signal.stft/frame/strided_slice/stack_1:output:0Lmelspectrogram/stft/stft_tf.signal.stft/frame/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2=
;melspectrogram/stft/stft_tf.signal.stft/frame/strided_sliceЌ
3melspectrogram/stft/stft_tf.signal.stft/frame/sub/yConst*
_output_shapes
: *
dtype0*
value	B :25
3melspectrogram/stft/stft_tf.signal.stft/frame/sub/y
1melspectrogram/stft/stft_tf.signal.stft/frame/subSub;melspectrogram/stft/stft_tf.signal.stft/frame/Rank:output:0<melspectrogram/stft/stft_tf.signal.stft/frame/sub/y:output:0*
T0*
_output_shapes
: 23
1melspectrogram/stft/stft_tf.signal.stft/frame/sub
3melspectrogram/stft/stft_tf.signal.stft/frame/sub_1Sub5melspectrogram/stft/stft_tf.signal.stft/frame/sub:z:0Dmelspectrogram/stft/stft_tf.signal.stft/frame/strided_slice:output:0*
T0*
_output_shapes
: 25
3melspectrogram/stft/stft_tf.signal.stft/frame/sub_1В
6melspectrogram/stft/stft_tf.signal.stft/frame/packed/1Const*
_output_shapes
: *
dtype0*
value	B :28
6melspectrogram/stft/stft_tf.signal.stft/frame/packed/1т
4melspectrogram/stft/stft_tf.signal.stft/frame/packedPackDmelspectrogram/stft/stft_tf.signal.stft/frame/strided_slice:output:0?melspectrogram/stft/stft_tf.signal.stft/frame/packed/1:output:07melspectrogram/stft/stft_tf.signal.stft/frame/sub_1:z:0*
N*
T0*
_output_shapes
:26
4melspectrogram/stft/stft_tf.signal.stft/frame/packedР
=melspectrogram/stft/stft_tf.signal.stft/frame/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2?
=melspectrogram/stft/stft_tf.signal.stft/frame/split/split_dim
3melspectrogram/stft/stft_tf.signal.stft/frame/splitSplitV<melspectrogram/stft/stft_tf.signal.stft/frame/Shape:output:0=melspectrogram/stft/stft_tf.signal.stft/frame/packed:output:0Fmelspectrogram/stft/stft_tf.signal.stft/frame/split/split_dim:output:0*
T0*

Tlen0*$
_output_shapes
::: *
	num_split25
3melspectrogram/stft/stft_tf.signal.stft/frame/splitН
;melspectrogram/stft/stft_tf.signal.stft/frame/Reshape/shapeConst*
_output_shapes
: *
dtype0*
valueB 2=
;melspectrogram/stft/stft_tf.signal.stft/frame/Reshape/shapeС
=melspectrogram/stft/stft_tf.signal.stft/frame/Reshape/shape_1Const*
_output_shapes
: *
dtype0*
valueB 2?
=melspectrogram/stft/stft_tf.signal.stft/frame/Reshape/shape_1 
5melspectrogram/stft/stft_tf.signal.stft/frame/ReshapeReshape<melspectrogram/stft/stft_tf.signal.stft/frame/split:output:1Fmelspectrogram/stft/stft_tf.signal.stft/frame/Reshape/shape_1:output:0*
T0*
_output_shapes
: 27
5melspectrogram/stft/stft_tf.signal.stft/frame/ReshapeЊ
2melspectrogram/stft/stft_tf.signal.stft/frame/SizeConst*
_output_shapes
: *
dtype0*
value	B :24
2melspectrogram/stft/stft_tf.signal.stft/frame/SizeЎ
4melspectrogram/stft/stft_tf.signal.stft/frame/Size_1Const*
_output_shapes
: *
dtype0*
value	B : 26
4melspectrogram/stft/stft_tf.signal.stft/frame/Size_1
3melspectrogram/stft/stft_tf.signal.stft/frame/sub_2Sub>melspectrogram/stft/stft_tf.signal.stft/frame/Reshape:output:0=melspectrogram/stft/stft_tf.signal.stft/frame_length:output:0*
T0*
_output_shapes
: 25
3melspectrogram/stft/stft_tf.signal.stft/frame/sub_2
6melspectrogram/stft/stft_tf.signal.stft/frame/floordivFloorDiv7melspectrogram/stft/stft_tf.signal.stft/frame/sub_2:z:0;melspectrogram/stft/stft_tf.signal.stft/frame_step:output:0*
T0*
_output_shapes
: 28
6melspectrogram/stft/stft_tf.signal.stft/frame/floordivЌ
3melspectrogram/stft/stft_tf.signal.stft/frame/add/xConst*
_output_shapes
: *
dtype0*
value	B :25
3melspectrogram/stft/stft_tf.signal.stft/frame/add/x
1melspectrogram/stft/stft_tf.signal.stft/frame/addAddV2<melspectrogram/stft/stft_tf.signal.stft/frame/add/x:output:0:melspectrogram/stft/stft_tf.signal.stft/frame/floordiv:z:0*
T0*
_output_shapes
: 23
1melspectrogram/stft/stft_tf.signal.stft/frame/addД
7melspectrogram/stft/stft_tf.signal.stft/frame/Maximum/xConst*
_output_shapes
: *
dtype0*
value	B : 29
7melspectrogram/stft/stft_tf.signal.stft/frame/Maximum/x
5melspectrogram/stft/stft_tf.signal.stft/frame/MaximumMaximum@melspectrogram/stft/stft_tf.signal.stft/frame/Maximum/x:output:05melspectrogram/stft/stft_tf.signal.stft/frame/add:z:0*
T0*
_output_shapes
: 27
5melspectrogram/stft/stft_tf.signal.stft/frame/MaximumЕ
7melspectrogram/stft/stft_tf.signal.stft/frame/gcd/ConstConst*
_output_shapes
: *
dtype0*
value
B :29
7melspectrogram/stft/stft_tf.signal.stft/frame/gcd/ConstЛ
:melspectrogram/stft/stft_tf.signal.stft/frame/floordiv_1/yConst*
_output_shapes
: *
dtype0*
value
B :2<
:melspectrogram/stft/stft_tf.signal.stft/frame/floordiv_1/yЅ
8melspectrogram/stft/stft_tf.signal.stft/frame/floordiv_1FloorDiv=melspectrogram/stft/stft_tf.signal.stft/frame_length:output:0Cmelspectrogram/stft/stft_tf.signal.stft/frame/floordiv_1/y:output:0*
T0*
_output_shapes
: 2:
8melspectrogram/stft/stft_tf.signal.stft/frame/floordiv_1Л
:melspectrogram/stft/stft_tf.signal.stft/frame/floordiv_2/yConst*
_output_shapes
: *
dtype0*
value
B :2<
:melspectrogram/stft/stft_tf.signal.stft/frame/floordiv_2/yЃ
8melspectrogram/stft/stft_tf.signal.stft/frame/floordiv_2FloorDiv;melspectrogram/stft/stft_tf.signal.stft/frame_step:output:0Cmelspectrogram/stft/stft_tf.signal.stft/frame/floordiv_2/y:output:0*
T0*
_output_shapes
: 2:
8melspectrogram/stft/stft_tf.signal.stft/frame/floordiv_2Л
:melspectrogram/stft/stft_tf.signal.stft/frame/floordiv_3/yConst*
_output_shapes
: *
dtype0*
value
B :2<
:melspectrogram/stft/stft_tf.signal.stft/frame/floordiv_3/yІ
8melspectrogram/stft/stft_tf.signal.stft/frame/floordiv_3FloorDiv>melspectrogram/stft/stft_tf.signal.stft/frame/Reshape:output:0Cmelspectrogram/stft/stft_tf.signal.stft/frame/floordiv_3/y:output:0*
T0*
_output_shapes
: 2:
8melspectrogram/stft/stft_tf.signal.stft/frame/floordiv_3­
3melspectrogram/stft/stft_tf.signal.stft/frame/mul/yConst*
_output_shapes
: *
dtype0*
value
B :25
3melspectrogram/stft/stft_tf.signal.stft/frame/mul/y
1melspectrogram/stft/stft_tf.signal.stft/frame/mulMul<melspectrogram/stft/stft_tf.signal.stft/frame/floordiv_3:z:0<melspectrogram/stft/stft_tf.signal.stft/frame/mul/y:output:0*
T0*
_output_shapes
: 23
1melspectrogram/stft/stft_tf.signal.stft/frame/mulы
=melspectrogram/stft/stft_tf.signal.stft/frame/concat/values_1Pack5melspectrogram/stft/stft_tf.signal.stft/frame/mul:z:0*
N*
T0*
_output_shapes
:2?
=melspectrogram/stft/stft_tf.signal.stft/frame/concat/values_1И
9melspectrogram/stft/stft_tf.signal.stft/frame/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2;
9melspectrogram/stft/stft_tf.signal.stft/frame/concat/axisЎ
4melspectrogram/stft/stft_tf.signal.stft/frame/concatConcatV2<melspectrogram/stft/stft_tf.signal.stft/frame/split:output:0Fmelspectrogram/stft/stft_tf.signal.stft/frame/concat/values_1:output:0<melspectrogram/stft/stft_tf.signal.stft/frame/split:output:2Bmelspectrogram/stft/stft_tf.signal.stft/frame/concat/axis:output:0*
N*
T0*
_output_shapes
:26
4melspectrogram/stft/stft_tf.signal.stft/frame/concatЩ
Amelspectrogram/stft/stft_tf.signal.stft/frame/concat_1/values_1/1Const*
_output_shapes
: *
dtype0*
value
B :2C
Amelspectrogram/stft/stft_tf.signal.stft/frame/concat_1/values_1/1Т
?melspectrogram/stft/stft_tf.signal.stft/frame/concat_1/values_1Pack<melspectrogram/stft/stft_tf.signal.stft/frame/floordiv_3:z:0Jmelspectrogram/stft/stft_tf.signal.stft/frame/concat_1/values_1/1:output:0*
N*
T0*
_output_shapes
:2A
?melspectrogram/stft/stft_tf.signal.stft/frame/concat_1/values_1М
;melspectrogram/stft/stft_tf.signal.stft/frame/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2=
;melspectrogram/stft/stft_tf.signal.stft/frame/concat_1/axisЖ
6melspectrogram/stft/stft_tf.signal.stft/frame/concat_1ConcatV2<melspectrogram/stft/stft_tf.signal.stft/frame/split:output:0Hmelspectrogram/stft/stft_tf.signal.stft/frame/concat_1/values_1:output:0<melspectrogram/stft/stft_tf.signal.stft/frame/split:output:2Dmelspectrogram/stft/stft_tf.signal.stft/frame/concat_1/axis:output:0*
N*
T0*
_output_shapes
:28
6melspectrogram/stft/stft_tf.signal.stft/frame/concat_1О
8melspectrogram/stft/stft_tf.signal.stft/frame/zeros_likeConst*
_output_shapes
:*
dtype0*
valueB: 2:
8melspectrogram/stft/stft_tf.signal.stft/frame/zeros_likeШ
=melspectrogram/stft/stft_tf.signal.stft/frame/ones_like/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2?
=melspectrogram/stft/stft_tf.signal.stft/frame/ones_like/ShapeР
=melspectrogram/stft/stft_tf.signal.stft/frame/ones_like/ConstConst*
_output_shapes
: *
dtype0*
value	B :2?
=melspectrogram/stft/stft_tf.signal.stft/frame/ones_like/ConstЏ
7melspectrogram/stft/stft_tf.signal.stft/frame/ones_likeFillFmelspectrogram/stft/stft_tf.signal.stft/frame/ones_like/Shape:output:0Fmelspectrogram/stft/stft_tf.signal.stft/frame/ones_like/Const:output:0*
T0*
_output_shapes
:29
7melspectrogram/stft/stft_tf.signal.stft/frame/ones_likeФ
:melspectrogram/stft/stft_tf.signal.stft/frame/StridedSliceStridedSlice!melspectrogram/stft/transpose:y:0Amelspectrogram/stft/stft_tf.signal.stft/frame/zeros_like:output:0=melspectrogram/stft/stft_tf.signal.stft/frame/concat:output:0@melspectrogram/stft/stft_tf.signal.stft/frame/ones_like:output:0*
Index0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ2<
:melspectrogram/stft/stft_tf.signal.stft/frame/StridedSliceа
7melspectrogram/stft/stft_tf.signal.stft/frame/Reshape_1ReshapeCmelspectrogram/stft/stft_tf.signal.stft/frame/StridedSlice:output:0?melspectrogram/stft/stft_tf.signal.stft/frame/concat_1:output:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ29
7melspectrogram/stft/stft_tf.signal.stft/frame/Reshape_1М
;melspectrogram/stft/stft_tf.signal.stft/frame/range_1/startConst*
_output_shapes
: *
dtype0*
value	B : 2=
;melspectrogram/stft/stft_tf.signal.stft/frame/range_1/startМ
;melspectrogram/stft/stft_tf.signal.stft/frame/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :2=
;melspectrogram/stft/stft_tf.signal.stft/frame/range_1/deltaу
5melspectrogram/stft/stft_tf.signal.stft/frame/range_1RangeDmelspectrogram/stft/stft_tf.signal.stft/frame/range_1/start:output:09melspectrogram/stft/stft_tf.signal.stft/frame/Maximum:z:0Dmelspectrogram/stft/stft_tf.signal.stft/frame/range_1/delta:output:0*#
_output_shapes
:џџџџџџџџџ27
5melspectrogram/stft/stft_tf.signal.stft/frame/range_1
3melspectrogram/stft/stft_tf.signal.stft/frame/mul_1Mul>melspectrogram/stft/stft_tf.signal.stft/frame/range_1:output:0<melspectrogram/stft/stft_tf.signal.stft/frame/floordiv_2:z:0*
T0*#
_output_shapes
:џџџџџџџџџ25
3melspectrogram/stft/stft_tf.signal.stft/frame/mul_1Ф
?melspectrogram/stft/stft_tf.signal.stft/frame/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2A
?melspectrogram/stft/stft_tf.signal.stft/frame/Reshape_2/shape/1Й
=melspectrogram/stft/stft_tf.signal.stft/frame/Reshape_2/shapePack9melspectrogram/stft/stft_tf.signal.stft/frame/Maximum:z:0Hmelspectrogram/stft/stft_tf.signal.stft/frame/Reshape_2/shape/1:output:0*
N*
T0*
_output_shapes
:2?
=melspectrogram/stft/stft_tf.signal.stft/frame/Reshape_2/shapeА
7melspectrogram/stft/stft_tf.signal.stft/frame/Reshape_2Reshape7melspectrogram/stft/stft_tf.signal.stft/frame/mul_1:z:0Fmelspectrogram/stft/stft_tf.signal.stft/frame/Reshape_2/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ29
7melspectrogram/stft/stft_tf.signal.stft/frame/Reshape_2М
;melspectrogram/stft/stft_tf.signal.stft/frame/range_2/startConst*
_output_shapes
: *
dtype0*
value	B : 2=
;melspectrogram/stft/stft_tf.signal.stft/frame/range_2/startМ
;melspectrogram/stft/stft_tf.signal.stft/frame/range_2/deltaConst*
_output_shapes
: *
dtype0*
value	B :2=
;melspectrogram/stft/stft_tf.signal.stft/frame/range_2/deltaн
5melspectrogram/stft/stft_tf.signal.stft/frame/range_2RangeDmelspectrogram/stft/stft_tf.signal.stft/frame/range_2/start:output:0<melspectrogram/stft/stft_tf.signal.stft/frame/floordiv_1:z:0Dmelspectrogram/stft/stft_tf.signal.stft/frame/range_2/delta:output:0*
_output_shapes
:27
5melspectrogram/stft/stft_tf.signal.stft/frame/range_2Ф
?melspectrogram/stft/stft_tf.signal.stft/frame/Reshape_3/shape/0Const*
_output_shapes
: *
dtype0*
value	B :2A
?melspectrogram/stft/stft_tf.signal.stft/frame/Reshape_3/shape/0М
=melspectrogram/stft/stft_tf.signal.stft/frame/Reshape_3/shapePackHmelspectrogram/stft/stft_tf.signal.stft/frame/Reshape_3/shape/0:output:0<melspectrogram/stft/stft_tf.signal.stft/frame/floordiv_1:z:0*
N*
T0*
_output_shapes
:2?
=melspectrogram/stft/stft_tf.signal.stft/frame/Reshape_3/shapeЎ
7melspectrogram/stft/stft_tf.signal.stft/frame/Reshape_3Reshape>melspectrogram/stft/stft_tf.signal.stft/frame/range_2:output:0Fmelspectrogram/stft/stft_tf.signal.stft/frame/Reshape_3/shape:output:0*
T0*
_output_shapes

:29
7melspectrogram/stft/stft_tf.signal.stft/frame/Reshape_3Љ
3melspectrogram/stft/stft_tf.signal.stft/frame/add_1AddV2@melspectrogram/stft/stft_tf.signal.stft/frame/Reshape_2:output:0@melspectrogram/stft/stft_tf.signal.stft/frame/Reshape_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ25
3melspectrogram/stft/stft_tf.signal.stft/frame/add_1Б
6melspectrogram/stft/stft_tf.signal.stft/frame/GatherV2GatherV2@melspectrogram/stft/stft_tf.signal.stft/frame/Reshape_1:output:07melspectrogram/stft/stft_tf.signal.stft/frame/add_1:z:0Dmelspectrogram/stft/stft_tf.signal.stft/frame/strided_slice:output:0*
Taxis0*
Tindices0*
Tparams0*F
_output_shapes4
2:0џџџџџџџџџџџџџџџџџџџџџџџџџџџ28
6melspectrogram/stft/stft_tf.signal.stft/frame/GatherV2В
?melspectrogram/stft/stft_tf.signal.stft/frame/concat_2/values_1Pack9melspectrogram/stft/stft_tf.signal.stft/frame/Maximum:z:0=melspectrogram/stft/stft_tf.signal.stft/frame_length:output:0*
N*
T0*
_output_shapes
:2A
?melspectrogram/stft/stft_tf.signal.stft/frame/concat_2/values_1М
;melspectrogram/stft/stft_tf.signal.stft/frame/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2=
;melspectrogram/stft/stft_tf.signal.stft/frame/concat_2/axisЖ
6melspectrogram/stft/stft_tf.signal.stft/frame/concat_2ConcatV2<melspectrogram/stft/stft_tf.signal.stft/frame/split:output:0Hmelspectrogram/stft/stft_tf.signal.stft/frame/concat_2/values_1:output:0<melspectrogram/stft/stft_tf.signal.stft/frame/split:output:2Dmelspectrogram/stft/stft_tf.signal.stft/frame/concat_2/axis:output:0*
N*
T0*
_output_shapes
:28
6melspectrogram/stft/stft_tf.signal.stft/frame/concat_2Л
7melspectrogram/stft/stft_tf.signal.stft/frame/Reshape_4Reshape?melspectrogram/stft/stft_tf.signal.stft/frame/GatherV2:output:0?melspectrogram/stft/stft_tf.signal.stft/frame/concat_2:output:0*
T0*1
_output_shapes
:џџџџџџџџџЗ29
7melspectrogram/stft/stft_tf.signal.stft/frame/Reshape_4О
<melspectrogram/stft/stft_tf.signal.stft/hann_window/periodicConst*
_output_shapes
: *
dtype0
*
value	B
 Z2>
<melspectrogram/stft/stft_tf.signal.stft/hann_window/periodicѓ
8melspectrogram/stft/stft_tf.signal.stft/hann_window/CastCastEmelspectrogram/stft/stft_tf.signal.stft/hann_window/periodic:output:0*

DstT0*

SrcT0
*
_output_shapes
: 2:
8melspectrogram/stft/stft_tf.signal.stft/hann_window/CastТ
>melspectrogram/stft/stft_tf.signal.stft/hann_window/FloorMod/yConst*
_output_shapes
: *
dtype0*
value	B :2@
>melspectrogram/stft/stft_tf.signal.stft/hann_window/FloorMod/yБ
<melspectrogram/stft/stft_tf.signal.stft/hann_window/FloorModFloorMod=melspectrogram/stft/stft_tf.signal.stft/frame_length:output:0Gmelspectrogram/stft/stft_tf.signal.stft/hann_window/FloorMod/y:output:0*
T0*
_output_shapes
: 2>
<melspectrogram/stft/stft_tf.signal.stft/hann_window/FloorModИ
9melspectrogram/stft/stft_tf.signal.stft/hann_window/sub/xConst*
_output_shapes
: *
dtype0*
value	B :2;
9melspectrogram/stft/stft_tf.signal.stft/hann_window/sub/x 
7melspectrogram/stft/stft_tf.signal.stft/hann_window/subSubBmelspectrogram/stft/stft_tf.signal.stft/hann_window/sub/x:output:0@melspectrogram/stft/stft_tf.signal.stft/hann_window/FloorMod:z:0*
T0*
_output_shapes
: 29
7melspectrogram/stft/stft_tf.signal.stft/hann_window/sub
7melspectrogram/stft/stft_tf.signal.stft/hann_window/mulMul<melspectrogram/stft/stft_tf.signal.stft/hann_window/Cast:y:0;melspectrogram/stft/stft_tf.signal.stft/hann_window/sub:z:0*
T0*
_output_shapes
: 29
7melspectrogram/stft/stft_tf.signal.stft/hann_window/mul
7melspectrogram/stft/stft_tf.signal.stft/hann_window/addAddV2=melspectrogram/stft/stft_tf.signal.stft/frame_length:output:0;melspectrogram/stft/stft_tf.signal.stft/hann_window/mul:z:0*
T0*
_output_shapes
: 29
7melspectrogram/stft/stft_tf.signal.stft/hann_window/addМ
;melspectrogram/stft/stft_tf.signal.stft/hann_window/sub_1/yConst*
_output_shapes
: *
dtype0*
value	B :2=
;melspectrogram/stft/stft_tf.signal.stft/hann_window/sub_1/yЁ
9melspectrogram/stft/stft_tf.signal.stft/hann_window/sub_1Sub;melspectrogram/stft/stft_tf.signal.stft/hann_window/add:z:0Dmelspectrogram/stft/stft_tf.signal.stft/hann_window/sub_1/y:output:0*
T0*
_output_shapes
: 2;
9melspectrogram/stft/stft_tf.signal.stft/hann_window/sub_1я
:melspectrogram/stft/stft_tf.signal.stft/hann_window/Cast_1Cast=melspectrogram/stft/stft_tf.signal.stft/hann_window/sub_1:z:0*

DstT0*

SrcT0*
_output_shapes
: 2<
:melspectrogram/stft/stft_tf.signal.stft/hann_window/Cast_1Ф
?melspectrogram/stft/stft_tf.signal.stft/hann_window/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2A
?melspectrogram/stft/stft_tf.signal.stft/hann_window/range/startФ
?melspectrogram/stft/stft_tf.signal.stft/hann_window/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2A
?melspectrogram/stft/stft_tf.signal.stft/hann_window/range/deltaя
9melspectrogram/stft/stft_tf.signal.stft/hann_window/rangeRangeHmelspectrogram/stft/stft_tf.signal.stft/hann_window/range/start:output:0=melspectrogram/stft/stft_tf.signal.stft/frame_length:output:0Hmelspectrogram/stft/stft_tf.signal.stft/hann_window/range/delta:output:0*
_output_shapes	
:2;
9melspectrogram/stft/stft_tf.signal.stft/hann_window/rangeљ
:melspectrogram/stft/stft_tf.signal.stft/hann_window/Cast_2CastBmelspectrogram/stft/stft_tf.signal.stft/hann_window/range:output:0*

DstT0*

SrcT0*
_output_shapes	
:2<
:melspectrogram/stft/stft_tf.signal.stft/hann_window/Cast_2Л
9melspectrogram/stft/stft_tf.signal.stft/hann_window/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *лЩ@2;
9melspectrogram/stft/stft_tf.signal.stft/hann_window/ConstЇ
9melspectrogram/stft/stft_tf.signal.stft/hann_window/mul_1MulBmelspectrogram/stft/stft_tf.signal.stft/hann_window/Const:output:0>melspectrogram/stft/stft_tf.signal.stft/hann_window/Cast_2:y:0*
T0*
_output_shapes	
:2;
9melspectrogram/stft/stft_tf.signal.stft/hann_window/mul_1Њ
;melspectrogram/stft/stft_tf.signal.stft/hann_window/truedivRealDiv=melspectrogram/stft/stft_tf.signal.stft/hann_window/mul_1:z:0>melspectrogram/stft/stft_tf.signal.stft/hann_window/Cast_1:y:0*
T0*
_output_shapes	
:2=
;melspectrogram/stft/stft_tf.signal.stft/hann_window/truedivр
7melspectrogram/stft/stft_tf.signal.stft/hann_window/CosCos?melspectrogram/stft/stft_tf.signal.stft/hann_window/truediv:z:0*
T0*
_output_shapes	
:29
7melspectrogram/stft/stft_tf.signal.stft/hann_window/CosП
;melspectrogram/stft/stft_tf.signal.stft/hann_window/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2=
;melspectrogram/stft/stft_tf.signal.stft/hann_window/mul_2/xІ
9melspectrogram/stft/stft_tf.signal.stft/hann_window/mul_2MulDmelspectrogram/stft/stft_tf.signal.stft/hann_window/mul_2/x:output:0;melspectrogram/stft/stft_tf.signal.stft/hann_window/Cos:y:0*
T0*
_output_shapes	
:2;
9melspectrogram/stft/stft_tf.signal.stft/hann_window/mul_2П
;melspectrogram/stft/stft_tf.signal.stft/hann_window/sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2=
;melspectrogram/stft/stft_tf.signal.stft/hann_window/sub_2/xЈ
9melspectrogram/stft/stft_tf.signal.stft/hann_window/sub_2SubDmelspectrogram/stft/stft_tf.signal.stft/hann_window/sub_2/x:output:0=melspectrogram/stft/stft_tf.signal.stft/hann_window/mul_2:z:0*
T0*
_output_shapes	
:2;
9melspectrogram/stft/stft_tf.signal.stft/hann_window/sub_2
+melspectrogram/stft/stft_tf.signal.stft/mulMul@melspectrogram/stft/stft_tf.signal.stft/frame/Reshape_4:output:0=melspectrogram/stft/stft_tf.signal.stft/hann_window/sub_2:z:0*
T0*1
_output_shapes
:џџџџџџџџџЗ2-
+melspectrogram/stft/stft_tf.signal.stft/mulн
3melspectrogram/stft/stft_tf.signal.stft/rfft/packedPack;melspectrogram/stft/stft_tf.signal.stft/fft_length:output:0*
N*
T0*
_output_shapes
:25
3melspectrogram/stft/stft_tf.signal.stft/rfft/packedН
7melspectrogram/stft/stft_tf.signal.stft/rfft/fft_lengthConst*
_output_shapes
:*
dtype0*
valueB:29
7melspectrogram/stft/stft_tf.signal.stft/rfft/fft_length
,melspectrogram/stft/stft_tf.signal.stft/rfftRFFT/melspectrogram/stft/stft_tf.signal.stft/mul:z:0@melspectrogram/stft/stft_tf.signal.stft/rfft/fft_length:output:0*1
_output_shapes
:џџџџџџџџџЗ2.
,melspectrogram/stft/stft_tf.signal.stft/rfftЅ
$melspectrogram/stft/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             2&
$melspectrogram/stft/transpose_1/permё
melspectrogram/stft/transpose_1	Transpose5melspectrogram/stft/stft_tf.signal.stft/rfft:output:0-melspectrogram/stft/transpose_1/perm:output:0*
T0*1
_output_shapes
:џџџџџџџџџЗ2!
melspectrogram/stft/transpose_1Ђ
melspectrogram/magnitude/Abs
ComplexAbs#melspectrogram/stft/transpose_1:y:0*1
_output_shapes
:џџџџџџџџџЗ2
melspectrogram/magnitude/AbsЊ
.melspectrogram/apply_filterbank/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:20
.melspectrogram/apply_filterbank/Tensordot/axesЕ
.melspectrogram/apply_filterbank/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          20
.melspectrogram/apply_filterbank/Tensordot/freeВ
/melspectrogram/apply_filterbank/Tensordot/ShapeShape melspectrogram/magnitude/Abs:y:0*
T0*
_output_shapes
:21
/melspectrogram/apply_filterbank/Tensordot/ShapeД
7melspectrogram/apply_filterbank/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 29
7melspectrogram/apply_filterbank/Tensordot/GatherV2/axisё
2melspectrogram/apply_filterbank/Tensordot/GatherV2GatherV28melspectrogram/apply_filterbank/Tensordot/Shape:output:07melspectrogram/apply_filterbank/Tensordot/free:output:0@melspectrogram/apply_filterbank/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:24
2melspectrogram/apply_filterbank/Tensordot/GatherV2И
9melspectrogram/apply_filterbank/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2;
9melspectrogram/apply_filterbank/Tensordot/GatherV2_1/axisї
4melspectrogram/apply_filterbank/Tensordot/GatherV2_1GatherV28melspectrogram/apply_filterbank/Tensordot/Shape:output:07melspectrogram/apply_filterbank/Tensordot/axes:output:0Bmelspectrogram/apply_filterbank/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:26
4melspectrogram/apply_filterbank/Tensordot/GatherV2_1Ќ
/melspectrogram/apply_filterbank/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 21
/melspectrogram/apply_filterbank/Tensordot/Const
.melspectrogram/apply_filterbank/Tensordot/ProdProd;melspectrogram/apply_filterbank/Tensordot/GatherV2:output:08melspectrogram/apply_filterbank/Tensordot/Const:output:0*
T0*
_output_shapes
: 20
.melspectrogram/apply_filterbank/Tensordot/ProdА
1melspectrogram/apply_filterbank/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 23
1melspectrogram/apply_filterbank/Tensordot/Const_1
0melspectrogram/apply_filterbank/Tensordot/Prod_1Prod=melspectrogram/apply_filterbank/Tensordot/GatherV2_1:output:0:melspectrogram/apply_filterbank/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 22
0melspectrogram/apply_filterbank/Tensordot/Prod_1А
5melspectrogram/apply_filterbank/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 27
5melspectrogram/apply_filterbank/Tensordot/concat/axisа
0melspectrogram/apply_filterbank/Tensordot/concatConcatV27melspectrogram/apply_filterbank/Tensordot/free:output:07melspectrogram/apply_filterbank/Tensordot/axes:output:0>melspectrogram/apply_filterbank/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:22
0melspectrogram/apply_filterbank/Tensordot/concat
/melspectrogram/apply_filterbank/Tensordot/stackPack7melspectrogram/apply_filterbank/Tensordot/Prod:output:09melspectrogram/apply_filterbank/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:21
/melspectrogram/apply_filterbank/Tensordot/stack
3melspectrogram/apply_filterbank/Tensordot/transpose	Transpose melspectrogram/magnitude/Abs:y:09melspectrogram/apply_filterbank/Tensordot/concat:output:0*
T0*1
_output_shapes
:џџџџџџџџџЗ25
3melspectrogram/apply_filterbank/Tensordot/transpose
1melspectrogram/apply_filterbank/Tensordot/ReshapeReshape7melspectrogram/apply_filterbank/Tensordot/transpose:y:08melspectrogram/apply_filterbank/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ23
1melspectrogram/apply_filterbank/Tensordot/Reshape
0melspectrogram/apply_filterbank/Tensordot/MatMulMatMul:melspectrogram/apply_filterbank/Tensordot/Reshape:output:0+melspectrogram_apply_filterbank_tensordot_b*
T0*(
_output_shapes
:џџџџџџџџџ22
0melspectrogram/apply_filterbank/Tensordot/MatMulБ
1melspectrogram/apply_filterbank/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:23
1melspectrogram/apply_filterbank/Tensordot/Const_2Д
7melspectrogram/apply_filterbank/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 29
7melspectrogram/apply_filterbank/Tensordot/concat_1/axisн
2melspectrogram/apply_filterbank/Tensordot/concat_1ConcatV2;melspectrogram/apply_filterbank/Tensordot/GatherV2:output:0:melspectrogram/apply_filterbank/Tensordot/Const_2:output:0@melspectrogram/apply_filterbank/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:24
2melspectrogram/apply_filterbank/Tensordot/concat_1
)melspectrogram/apply_filterbank/TensordotReshape:melspectrogram/apply_filterbank/Tensordot/MatMul:product:0;melspectrogram/apply_filterbank/Tensordot/concat_1:output:0*
T0*1
_output_shapes
:џџџџџџџџџЗ2+
)melspectrogram/apply_filterbank/TensordotЙ
.melspectrogram/apply_filterbank/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             20
.melspectrogram/apply_filterbank/transpose/perm
)melspectrogram/apply_filterbank/transpose	Transpose2melspectrogram/apply_filterbank/Tensordot:output:07melspectrogram/apply_filterbank/transpose/perm:output:0*
T0*1
_output_shapes
:џџџџџџџџџЗ2+
)melspectrogram/apply_filterbank/transposeА
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
:*
dtype02$
"batch_normalization/ReadVariableOpЖ
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
:*
dtype02&
$batch_normalization/ReadVariableOp_1у
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization/FusedBatchNormV3/ReadVariableOpщ
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype027
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ы
$batch_normalization/FusedBatchNormV3FusedBatchNormV3-melspectrogram/apply_filterbank/transpose:y:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:џџџџџџџџџЗ:::::*
epsilon%o:*
is_training( 2&
$batch_normalization/FusedBatchNormV3Њ
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
conv2d/Conv2D/ReadVariableOpм
conv2d/Conv2DConv2D(batch_normalization/FusedBatchNormV3:y:0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџЗ*
paddingSAME*
strides
2
conv2d/Conv2DЁ
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv2d/BiasAdd/ReadVariableOpІ
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџЗ2
conv2d/BiasAddw
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*1
_output_shapes
:џџџџџџџџџЗ2
conv2d/ReluС
max_pooling2d/MaxPoolMaxPoolconv2d/Relu:activations:0*/
_output_shapes
:џџџџџџџџџg@*
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPoolЖ
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes
:*
dtype02&
$batch_normalization_1/ReadVariableOpМ
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes
:*
dtype02(
&batch_normalization_1/ReadVariableOp_1щ
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpя
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype029
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ц
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3max_pooling2d/MaxPool:output:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџg@:::::*
epsilon%o:*
is_training( 2(
&batch_normalization_1/FusedBatchNormV3А
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02 
conv2d_1/Conv2D/ReadVariableOpт
conv2d_1/Conv2DConv2D*batch_normalization_1/FusedBatchNormV3:y:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџg@*
paddingSAME*
strides
2
conv2d_1/Conv2DЇ
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_1/BiasAdd/ReadVariableOpЌ
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџg@2
conv2d_1/BiasAdd{
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџg@2
conv2d_1/ReluЧ
max_pooling2d_1/MaxPoolMaxPoolconv2d_1/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ" *
ksize
*
paddingVALID*
strides
2
max_pooling2d_1/MaxPoolЖ
$batch_normalization_2/ReadVariableOpReadVariableOp-batch_normalization_2_readvariableop_resource*
_output_shapes
:*
dtype02&
$batch_normalization_2/ReadVariableOpМ
&batch_normalization_2/ReadVariableOp_1ReadVariableOp/batch_normalization_2_readvariableop_1_resource*
_output_shapes
:*
dtype02(
&batch_normalization_2/ReadVariableOp_1щ
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpя
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype029
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ш
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3 max_pooling2d_1/MaxPool:output:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ" :::::*
epsilon%o:*
is_training( 2(
&batch_normalization_2/FusedBatchNormV3А
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02 
conv2d_2/Conv2D/ReadVariableOpт
conv2d_2/Conv2DConv2D*batch_normalization_2/FusedBatchNormV3:y:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ"  *
paddingSAME*
strides
2
conv2d_2/Conv2DЇ
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_2/BiasAdd/ReadVariableOpЌ
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ"  2
conv2d_2/BiasAdd{
conv2d_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ"  2
conv2d_2/ReluЧ
max_pooling2d_2/MaxPoolMaxPoolconv2d_2/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ *
ksize
*
paddingVALID*
strides
2
max_pooling2d_2/MaxPoolЖ
$batch_normalization_3/ReadVariableOpReadVariableOp-batch_normalization_3_readvariableop_resource*
_output_shapes
: *
dtype02&
$batch_normalization_3/ReadVariableOpМ
&batch_normalization_3/ReadVariableOp_1ReadVariableOp/batch_normalization_3_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&batch_normalization_3/ReadVariableOp_1щ
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype027
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpя
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype029
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ш
&batch_normalization_3/FusedBatchNormV3FusedBatchNormV3 max_pooling2d_2/MaxPool:output:0,batch_normalization_3/ReadVariableOp:value:0.batch_normalization_3/ReadVariableOp_1:value:0=batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ : : : : :*
epsilon%o:*
is_training( 2(
&batch_normalization_3/FusedBatchNormV3o
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ "  2
flatten/ConstЄ
flatten/ReshapeReshape*batch_normalization_3/FusedBatchNormV3:y:0flatten/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџD2
flatten/Reshape 
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	D@*
dtype02
dense/MatMul/ReadVariableOp
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
dense/MatMul
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
dense/BiasAdd/ReadVariableOp
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
dense/BiasAddj

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2

dense/Relu|
dropout/IdentityIdentitydense/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
dropout/IdentityЅ
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02
dense_1/MatMul/ReadVariableOp
dense_1/MatMulMatMuldropout/Identity:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_1/MatMulЄ
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_1/BiasAdd/ReadVariableOpЁ
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_1/BiasAddp
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_1/Relu
dropout_1/IdentityIdentitydense_1/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dropout_1/IdentityЅ
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_2/MatMul/ReadVariableOp 
dense_2/MatMulMatMuldropout_1/Identity:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_2/MatMulЄ
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_2/BiasAdd/ReadVariableOpЁ
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_2/BiasAddy
dense_2/SigmoidSigmoiddense_2/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_2/Sigmoidы	
IdentityIdentitydense_2/Sigmoid:y:04^batch_normalization/FusedBatchNormV3/ReadVariableOp6^batch_normalization/FusedBatchNormV3/ReadVariableOp_1#^batch_normalization/ReadVariableOp%^batch_normalization/ReadVariableOp_16^batch_normalization_1/FusedBatchNormV3/ReadVariableOp8^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_1/ReadVariableOp'^batch_normalization_1/ReadVariableOp_16^batch_normalization_2/FusedBatchNormV3/ReadVariableOp8^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_2/ReadVariableOp'^batch_normalization_2/ReadVariableOp_16^batch_normalization_3/FusedBatchNormV3/ReadVariableOp8^batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_3/ReadVariableOp'^batch_normalization_3/ReadVariableOp_1^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*І
_input_shapes
:џџџџџџџџџт	:
::::::::::::::::::::::::::::2j
3batch_normalization/FusedBatchNormV3/ReadVariableOp3batch_normalization/FusedBatchNormV3/ReadVariableOp2n
5batch_normalization/FusedBatchNormV3/ReadVariableOp_15batch_normalization/FusedBatchNormV3/ReadVariableOp_12H
"batch_normalization/ReadVariableOp"batch_normalization/ReadVariableOp2L
$batch_normalization/ReadVariableOp_1$batch_normalization/ReadVariableOp_12n
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp5batch_normalization_1/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_17batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_1/ReadVariableOp$batch_normalization_1/ReadVariableOp2P
&batch_normalization_1/ReadVariableOp_1&batch_normalization_1/ReadVariableOp_12n
5batch_normalization_2/FusedBatchNormV3/ReadVariableOp5batch_normalization_2/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_17batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_2/ReadVariableOp$batch_normalization_2/ReadVariableOp2P
&batch_normalization_2/ReadVariableOp_1&batch_normalization_2/ReadVariableOp_12n
5batch_normalization_3/FusedBatchNormV3/ReadVariableOp5batch_normalization_3/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_17batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_3/ReadVariableOp$batch_normalization_3/ReadVariableOp2P
&batch_normalization_3/ReadVariableOp_1&batch_normalization_3/ReadVariableOp_12>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp:Q M
)
_output_shapes
:џџџџџџџџџт	
 
_user_specified_nameinputs:&"
 
_output_shapes
:

ю	
л
B__inference_dense_2_layer_call_and_return_conditional_losses_23079

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
Sigmoid
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Р
ѓ
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_22577

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1м
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3ь
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
е
Ј
5__inference_batch_normalization_1_layer_call_fn_22667

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityЂStatefulPartitionedCallЂ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџg@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_206132
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:џџџџџџџџџg@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџg@::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџg@
 
_user_specified_nameinputs
э	
й
@__inference_dense_layer_call_and_return_conditional_losses_20876

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	D@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџD::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:џџџџџџџџџD
 
_user_specified_nameinputs
дЦ
н*
!__inference__traced_restore_23743
file_prefix.
*assignvariableop_batch_normalization_gamma/
+assignvariableop_1_batch_normalization_beta6
2assignvariableop_2_batch_normalization_moving_mean:
6assignvariableop_3_batch_normalization_moving_variance$
 assignvariableop_4_conv2d_kernel"
assignvariableop_5_conv2d_bias2
.assignvariableop_6_batch_normalization_1_gamma1
-assignvariableop_7_batch_normalization_1_beta8
4assignvariableop_8_batch_normalization_1_moving_mean<
8assignvariableop_9_batch_normalization_1_moving_variance'
#assignvariableop_10_conv2d_1_kernel%
!assignvariableop_11_conv2d_1_bias3
/assignvariableop_12_batch_normalization_2_gamma2
.assignvariableop_13_batch_normalization_2_beta9
5assignvariableop_14_batch_normalization_2_moving_mean=
9assignvariableop_15_batch_normalization_2_moving_variance'
#assignvariableop_16_conv2d_2_kernel%
!assignvariableop_17_conv2d_2_bias3
/assignvariableop_18_batch_normalization_3_gamma2
.assignvariableop_19_batch_normalization_3_beta9
5assignvariableop_20_batch_normalization_3_moving_mean=
9assignvariableop_21_batch_normalization_3_moving_variance$
 assignvariableop_22_dense_kernel"
assignvariableop_23_dense_bias&
"assignvariableop_24_dense_1_kernel$
 assignvariableop_25_dense_1_bias&
"assignvariableop_26_dense_2_kernel$
 assignvariableop_27_dense_2_bias!
assignvariableop_28_adam_iter#
assignvariableop_29_adam_beta_1#
assignvariableop_30_adam_beta_2"
assignvariableop_31_adam_decay*
&assignvariableop_32_adam_learning_rate
assignvariableop_33_total
assignvariableop_34_count
assignvariableop_35_total_1
assignvariableop_36_count_18
4assignvariableop_37_adam_batch_normalization_gamma_m7
3assignvariableop_38_adam_batch_normalization_beta_m,
(assignvariableop_39_adam_conv2d_kernel_m*
&assignvariableop_40_adam_conv2d_bias_m:
6assignvariableop_41_adam_batch_normalization_1_gamma_m9
5assignvariableop_42_adam_batch_normalization_1_beta_m.
*assignvariableop_43_adam_conv2d_1_kernel_m,
(assignvariableop_44_adam_conv2d_1_bias_m:
6assignvariableop_45_adam_batch_normalization_2_gamma_m9
5assignvariableop_46_adam_batch_normalization_2_beta_m.
*assignvariableop_47_adam_conv2d_2_kernel_m,
(assignvariableop_48_adam_conv2d_2_bias_m:
6assignvariableop_49_adam_batch_normalization_3_gamma_m9
5assignvariableop_50_adam_batch_normalization_3_beta_m+
'assignvariableop_51_adam_dense_kernel_m)
%assignvariableop_52_adam_dense_bias_m-
)assignvariableop_53_adam_dense_1_kernel_m+
'assignvariableop_54_adam_dense_1_bias_m-
)assignvariableop_55_adam_dense_2_kernel_m+
'assignvariableop_56_adam_dense_2_bias_m8
4assignvariableop_57_adam_batch_normalization_gamma_v7
3assignvariableop_58_adam_batch_normalization_beta_v,
(assignvariableop_59_adam_conv2d_kernel_v*
&assignvariableop_60_adam_conv2d_bias_v:
6assignvariableop_61_adam_batch_normalization_1_gamma_v9
5assignvariableop_62_adam_batch_normalization_1_beta_v.
*assignvariableop_63_adam_conv2d_1_kernel_v,
(assignvariableop_64_adam_conv2d_1_bias_v:
6assignvariableop_65_adam_batch_normalization_2_gamma_v9
5assignvariableop_66_adam_batch_normalization_2_beta_v.
*assignvariableop_67_adam_conv2d_2_kernel_v,
(assignvariableop_68_adam_conv2d_2_bias_v:
6assignvariableop_69_adam_batch_normalization_3_gamma_v9
5assignvariableop_70_adam_batch_normalization_3_beta_v+
'assignvariableop_71_adam_dense_kernel_v)
%assignvariableop_72_adam_dense_bias_v-
)assignvariableop_73_adam_dense_1_kernel_v+
'assignvariableop_74_adam_dense_1_bias_v-
)assignvariableop_75_adam_dense_2_kernel_v+
'assignvariableop_76_adam_dense_2_bias_v
identity_78ЂAssignVariableOpЂAssignVariableOp_1ЂAssignVariableOp_10ЂAssignVariableOp_11ЂAssignVariableOp_12ЂAssignVariableOp_13ЂAssignVariableOp_14ЂAssignVariableOp_15ЂAssignVariableOp_16ЂAssignVariableOp_17ЂAssignVariableOp_18ЂAssignVariableOp_19ЂAssignVariableOp_2ЂAssignVariableOp_20ЂAssignVariableOp_21ЂAssignVariableOp_22ЂAssignVariableOp_23ЂAssignVariableOp_24ЂAssignVariableOp_25ЂAssignVariableOp_26ЂAssignVariableOp_27ЂAssignVariableOp_28ЂAssignVariableOp_29ЂAssignVariableOp_3ЂAssignVariableOp_30ЂAssignVariableOp_31ЂAssignVariableOp_32ЂAssignVariableOp_33ЂAssignVariableOp_34ЂAssignVariableOp_35ЂAssignVariableOp_36ЂAssignVariableOp_37ЂAssignVariableOp_38ЂAssignVariableOp_39ЂAssignVariableOp_4ЂAssignVariableOp_40ЂAssignVariableOp_41ЂAssignVariableOp_42ЂAssignVariableOp_43ЂAssignVariableOp_44ЂAssignVariableOp_45ЂAssignVariableOp_46ЂAssignVariableOp_47ЂAssignVariableOp_48ЂAssignVariableOp_49ЂAssignVariableOp_5ЂAssignVariableOp_50ЂAssignVariableOp_51ЂAssignVariableOp_52ЂAssignVariableOp_53ЂAssignVariableOp_54ЂAssignVariableOp_55ЂAssignVariableOp_56ЂAssignVariableOp_57ЂAssignVariableOp_58ЂAssignVariableOp_59ЂAssignVariableOp_6ЂAssignVariableOp_60ЂAssignVariableOp_61ЂAssignVariableOp_62ЂAssignVariableOp_63ЂAssignVariableOp_64ЂAssignVariableOp_65ЂAssignVariableOp_66ЂAssignVariableOp_67ЂAssignVariableOp_68ЂAssignVariableOp_69ЂAssignVariableOp_7ЂAssignVariableOp_70ЂAssignVariableOp_71ЂAssignVariableOp_72ЂAssignVariableOp_73ЂAssignVariableOp_74ЂAssignVariableOp_75ЂAssignVariableOp_76ЂAssignVariableOp_8ЂAssignVariableOp_9+
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:N*
dtype0* *
value*B*NB5layer_with_weights-0/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-0/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-0/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names­
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:N*
dtype0*Б
valueЇBЄNB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesД
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Ю
_output_shapesЛ
И::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*\
dtypesR
P2N	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

IdentityЉ
AssignVariableOpAssignVariableOp*assignvariableop_batch_normalization_gammaIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1А
AssignVariableOp_1AssignVariableOp+assignvariableop_1_batch_normalization_betaIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2З
AssignVariableOp_2AssignVariableOp2assignvariableop_2_batch_normalization_moving_meanIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3Л
AssignVariableOp_3AssignVariableOp6assignvariableop_3_batch_normalization_moving_varianceIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4Ѕ
AssignVariableOp_4AssignVariableOp assignvariableop_4_conv2d_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5Ѓ
AssignVariableOp_5AssignVariableOpassignvariableop_5_conv2d_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6Г
AssignVariableOp_6AssignVariableOp.assignvariableop_6_batch_normalization_1_gammaIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7В
AssignVariableOp_7AssignVariableOp-assignvariableop_7_batch_normalization_1_betaIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8Й
AssignVariableOp_8AssignVariableOp4assignvariableop_8_batch_normalization_1_moving_meanIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9Н
AssignVariableOp_9AssignVariableOp8assignvariableop_9_batch_normalization_1_moving_varianceIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10Ћ
AssignVariableOp_10AssignVariableOp#assignvariableop_10_conv2d_1_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11Љ
AssignVariableOp_11AssignVariableOp!assignvariableop_11_conv2d_1_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12З
AssignVariableOp_12AssignVariableOp/assignvariableop_12_batch_normalization_2_gammaIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13Ж
AssignVariableOp_13AssignVariableOp.assignvariableop_13_batch_normalization_2_betaIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14Н
AssignVariableOp_14AssignVariableOp5assignvariableop_14_batch_normalization_2_moving_meanIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15С
AssignVariableOp_15AssignVariableOp9assignvariableop_15_batch_normalization_2_moving_varianceIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16Ћ
AssignVariableOp_16AssignVariableOp#assignvariableop_16_conv2d_2_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17Љ
AssignVariableOp_17AssignVariableOp!assignvariableop_17_conv2d_2_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18З
AssignVariableOp_18AssignVariableOp/assignvariableop_18_batch_normalization_3_gammaIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19Ж
AssignVariableOp_19AssignVariableOp.assignvariableop_19_batch_normalization_3_betaIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20Н
AssignVariableOp_20AssignVariableOp5assignvariableop_20_batch_normalization_3_moving_meanIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21С
AssignVariableOp_21AssignVariableOp9assignvariableop_21_batch_normalization_3_moving_varianceIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22Ј
AssignVariableOp_22AssignVariableOp assignvariableop_22_dense_kernelIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23І
AssignVariableOp_23AssignVariableOpassignvariableop_23_dense_biasIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24Њ
AssignVariableOp_24AssignVariableOp"assignvariableop_24_dense_1_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25Ј
AssignVariableOp_25AssignVariableOp assignvariableop_25_dense_1_biasIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26Њ
AssignVariableOp_26AssignVariableOp"assignvariableop_26_dense_2_kernelIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27Ј
AssignVariableOp_27AssignVariableOp assignvariableop_27_dense_2_biasIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_28Ѕ
AssignVariableOp_28AssignVariableOpassignvariableop_28_adam_iterIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29Ї
AssignVariableOp_29AssignVariableOpassignvariableop_29_adam_beta_1Identity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30Ї
AssignVariableOp_30AssignVariableOpassignvariableop_30_adam_beta_2Identity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31І
AssignVariableOp_31AssignVariableOpassignvariableop_31_adam_decayIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32Ў
AssignVariableOp_32AssignVariableOp&assignvariableop_32_adam_learning_rateIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33Ё
AssignVariableOp_33AssignVariableOpassignvariableop_33_totalIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34Ё
AssignVariableOp_34AssignVariableOpassignvariableop_34_countIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35Ѓ
AssignVariableOp_35AssignVariableOpassignvariableop_35_total_1Identity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36Ѓ
AssignVariableOp_36AssignVariableOpassignvariableop_36_count_1Identity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37М
AssignVariableOp_37AssignVariableOp4assignvariableop_37_adam_batch_normalization_gamma_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38Л
AssignVariableOp_38AssignVariableOp3assignvariableop_38_adam_batch_normalization_beta_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39А
AssignVariableOp_39AssignVariableOp(assignvariableop_39_adam_conv2d_kernel_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40Ў
AssignVariableOp_40AssignVariableOp&assignvariableop_40_adam_conv2d_bias_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41О
AssignVariableOp_41AssignVariableOp6assignvariableop_41_adam_batch_normalization_1_gamma_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42Н
AssignVariableOp_42AssignVariableOp5assignvariableop_42_adam_batch_normalization_1_beta_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43В
AssignVariableOp_43AssignVariableOp*assignvariableop_43_adam_conv2d_1_kernel_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44А
AssignVariableOp_44AssignVariableOp(assignvariableop_44_adam_conv2d_1_bias_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45О
AssignVariableOp_45AssignVariableOp6assignvariableop_45_adam_batch_normalization_2_gamma_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46Н
AssignVariableOp_46AssignVariableOp5assignvariableop_46_adam_batch_normalization_2_beta_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47В
AssignVariableOp_47AssignVariableOp*assignvariableop_47_adam_conv2d_2_kernel_mIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48А
AssignVariableOp_48AssignVariableOp(assignvariableop_48_adam_conv2d_2_bias_mIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49О
AssignVariableOp_49AssignVariableOp6assignvariableop_49_adam_batch_normalization_3_gamma_mIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50Н
AssignVariableOp_50AssignVariableOp5assignvariableop_50_adam_batch_normalization_3_beta_mIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51Џ
AssignVariableOp_51AssignVariableOp'assignvariableop_51_adam_dense_kernel_mIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52­
AssignVariableOp_52AssignVariableOp%assignvariableop_52_adam_dense_bias_mIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53Б
AssignVariableOp_53AssignVariableOp)assignvariableop_53_adam_dense_1_kernel_mIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54Џ
AssignVariableOp_54AssignVariableOp'assignvariableop_54_adam_dense_1_bias_mIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55Б
AssignVariableOp_55AssignVariableOp)assignvariableop_55_adam_dense_2_kernel_mIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56Џ
AssignVariableOp_56AssignVariableOp'assignvariableop_56_adam_dense_2_bias_mIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57М
AssignVariableOp_57AssignVariableOp4assignvariableop_57_adam_batch_normalization_gamma_vIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58Л
AssignVariableOp_58AssignVariableOp3assignvariableop_58_adam_batch_normalization_beta_vIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59А
AssignVariableOp_59AssignVariableOp(assignvariableop_59_adam_conv2d_kernel_vIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60Ў
AssignVariableOp_60AssignVariableOp&assignvariableop_60_adam_conv2d_bias_vIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61О
AssignVariableOp_61AssignVariableOp6assignvariableop_61_adam_batch_normalization_1_gamma_vIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62Н
AssignVariableOp_62AssignVariableOp5assignvariableop_62_adam_batch_normalization_1_beta_vIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63В
AssignVariableOp_63AssignVariableOp*assignvariableop_63_adam_conv2d_1_kernel_vIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64А
AssignVariableOp_64AssignVariableOp(assignvariableop_64_adam_conv2d_1_bias_vIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_64n
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:2
Identity_65О
AssignVariableOp_65AssignVariableOp6assignvariableop_65_adam_batch_normalization_2_gamma_vIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_65n
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:2
Identity_66Н
AssignVariableOp_66AssignVariableOp5assignvariableop_66_adam_batch_normalization_2_beta_vIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_66n
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:2
Identity_67В
AssignVariableOp_67AssignVariableOp*assignvariableop_67_adam_conv2d_2_kernel_vIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_67n
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:2
Identity_68А
AssignVariableOp_68AssignVariableOp(assignvariableop_68_adam_conv2d_2_bias_vIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_68n
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:2
Identity_69О
AssignVariableOp_69AssignVariableOp6assignvariableop_69_adam_batch_normalization_3_gamma_vIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_69n
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:2
Identity_70Н
AssignVariableOp_70AssignVariableOp5assignvariableop_70_adam_batch_normalization_3_beta_vIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_70n
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:2
Identity_71Џ
AssignVariableOp_71AssignVariableOp'assignvariableop_71_adam_dense_kernel_vIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_71n
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:2
Identity_72­
AssignVariableOp_72AssignVariableOp%assignvariableop_72_adam_dense_bias_vIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_72n
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:2
Identity_73Б
AssignVariableOp_73AssignVariableOp)assignvariableop_73_adam_dense_1_kernel_vIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_73n
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:2
Identity_74Џ
AssignVariableOp_74AssignVariableOp'assignvariableop_74_adam_dense_1_bias_vIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_74n
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:2
Identity_75Б
AssignVariableOp_75AssignVariableOp)assignvariableop_75_adam_dense_2_kernel_vIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_75n
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:2
Identity_76Џ
AssignVariableOp_76AssignVariableOp'assignvariableop_76_adam_dense_2_bias_vIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_769
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpќ
Identity_77Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_77я
Identity_78IdentityIdentity_77:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_78"#
identity_78Identity_78:output:0*Ы
_input_shapesЙ
Ж: :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
Ь

P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_20159

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityЂAssignNewValueЂAssignNewValue_1ЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:*
exponential_avg_factor%
з#<2
FusedBatchNormV3­
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValueЛ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Р
ѓ
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_20190

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1м
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3ь
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ТЬ

I__inference_melspectrogram_layer_call_and_return_conditional_losses_22377

inputs 
apply_filterbank_tensordot_b
identity
stft/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
stft/transpose/perm
stft/transpose	Transposeinputsstft/transpose/perm:output:0*
T0*-
_output_shapes
:џџџџџџџџџт	2
stft/transpose
%stft/stft_tf.signal.stft/frame_lengthConst*
_output_shapes
: *
dtype0*
value
B :2'
%stft/stft_tf.signal.stft/frame_length
#stft/stft_tf.signal.stft/frame_stepConst*
_output_shapes
: *
dtype0*
value
B :2%
#stft/stft_tf.signal.stft/frame_step
#stft/stft_tf.signal.stft/fft_lengthConst*
_output_shapes
: *
dtype0*
value
B :2%
#stft/stft_tf.signal.stft/fft_length
#stft/stft_tf.signal.stft/frame/axisConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2%
#stft/stft_tf.signal.stft/frame/axis
$stft/stft_tf.signal.stft/frame/ShapeShapestft/transpose:y:0*
T0*
_output_shapes
:2&
$stft/stft_tf.signal.stft/frame/Shape
#stft/stft_tf.signal.stft/frame/RankConst*
_output_shapes
: *
dtype0*
value	B :2%
#stft/stft_tf.signal.stft/frame/Rank
*stft/stft_tf.signal.stft/frame/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2,
*stft/stft_tf.signal.stft/frame/range/start
*stft/stft_tf.signal.stft/frame/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2,
*stft/stft_tf.signal.stft/frame/range/delta
$stft/stft_tf.signal.stft/frame/rangeRange3stft/stft_tf.signal.stft/frame/range/start:output:0,stft/stft_tf.signal.stft/frame/Rank:output:03stft/stft_tf.signal.stft/frame/range/delta:output:0*
_output_shapes
:2&
$stft/stft_tf.signal.stft/frame/rangeЛ
2stft/stft_tf.signal.stft/frame/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ24
2stft/stft_tf.signal.stft/frame/strided_slice/stackЖ
4stft/stft_tf.signal.stft/frame/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 26
4stft/stft_tf.signal.stft/frame/strided_slice/stack_1Ж
4stft/stft_tf.signal.stft/frame/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:26
4stft/stft_tf.signal.stft/frame/strided_slice/stack_2
,stft/stft_tf.signal.stft/frame/strided_sliceStridedSlice-stft/stft_tf.signal.stft/frame/range:output:0;stft/stft_tf.signal.stft/frame/strided_slice/stack:output:0=stft/stft_tf.signal.stft/frame/strided_slice/stack_1:output:0=stft/stft_tf.signal.stft/frame/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2.
,stft/stft_tf.signal.stft/frame/strided_slice
$stft/stft_tf.signal.stft/frame/sub/yConst*
_output_shapes
: *
dtype0*
value	B :2&
$stft/stft_tf.signal.stft/frame/sub/yЭ
"stft/stft_tf.signal.stft/frame/subSub,stft/stft_tf.signal.stft/frame/Rank:output:0-stft/stft_tf.signal.stft/frame/sub/y:output:0*
T0*
_output_shapes
: 2$
"stft/stft_tf.signal.stft/frame/subг
$stft/stft_tf.signal.stft/frame/sub_1Sub&stft/stft_tf.signal.stft/frame/sub:z:05stft/stft_tf.signal.stft/frame/strided_slice:output:0*
T0*
_output_shapes
: 2&
$stft/stft_tf.signal.stft/frame/sub_1
'stft/stft_tf.signal.stft/frame/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2)
'stft/stft_tf.signal.stft/frame/packed/1
%stft/stft_tf.signal.stft/frame/packedPack5stft/stft_tf.signal.stft/frame/strided_slice:output:00stft/stft_tf.signal.stft/frame/packed/1:output:0(stft/stft_tf.signal.stft/frame/sub_1:z:0*
N*
T0*
_output_shapes
:2'
%stft/stft_tf.signal.stft/frame/packedЂ
.stft/stft_tf.signal.stft/frame/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 20
.stft/stft_tf.signal.stft/frame/split/split_dimК
$stft/stft_tf.signal.stft/frame/splitSplitV-stft/stft_tf.signal.stft/frame/Shape:output:0.stft/stft_tf.signal.stft/frame/packed:output:07stft/stft_tf.signal.stft/frame/split/split_dim:output:0*
T0*

Tlen0*$
_output_shapes
::: *
	num_split2&
$stft/stft_tf.signal.stft/frame/split
,stft/stft_tf.signal.stft/frame/Reshape/shapeConst*
_output_shapes
: *
dtype0*
valueB 2.
,stft/stft_tf.signal.stft/frame/Reshape/shapeЃ
.stft/stft_tf.signal.stft/frame/Reshape/shape_1Const*
_output_shapes
: *
dtype0*
valueB 20
.stft/stft_tf.signal.stft/frame/Reshape/shape_1ф
&stft/stft_tf.signal.stft/frame/ReshapeReshape-stft/stft_tf.signal.stft/frame/split:output:17stft/stft_tf.signal.stft/frame/Reshape/shape_1:output:0*
T0*
_output_shapes
: 2(
&stft/stft_tf.signal.stft/frame/Reshape
#stft/stft_tf.signal.stft/frame/SizeConst*
_output_shapes
: *
dtype0*
value	B :2%
#stft/stft_tf.signal.stft/frame/Size
%stft/stft_tf.signal.stft/frame/Size_1Const*
_output_shapes
: *
dtype0*
value	B : 2'
%stft/stft_tf.signal.stft/frame/Size_1е
$stft/stft_tf.signal.stft/frame/sub_2Sub/stft/stft_tf.signal.stft/frame/Reshape:output:0.stft/stft_tf.signal.stft/frame_length:output:0*
T0*
_output_shapes
: 2&
$stft/stft_tf.signal.stft/frame/sub_2з
'stft/stft_tf.signal.stft/frame/floordivFloorDiv(stft/stft_tf.signal.stft/frame/sub_2:z:0,stft/stft_tf.signal.stft/frame_step:output:0*
T0*
_output_shapes
: 2)
'stft/stft_tf.signal.stft/frame/floordiv
$stft/stft_tf.signal.stft/frame/add/xConst*
_output_shapes
: *
dtype0*
value	B :2&
$stft/stft_tf.signal.stft/frame/add/xЮ
"stft/stft_tf.signal.stft/frame/addAddV2-stft/stft_tf.signal.stft/frame/add/x:output:0+stft/stft_tf.signal.stft/frame/floordiv:z:0*
T0*
_output_shapes
: 2$
"stft/stft_tf.signal.stft/frame/add
(stft/stft_tf.signal.stft/frame/Maximum/xConst*
_output_shapes
: *
dtype0*
value	B : 2*
(stft/stft_tf.signal.stft/frame/Maximum/xз
&stft/stft_tf.signal.stft/frame/MaximumMaximum1stft/stft_tf.signal.stft/frame/Maximum/x:output:0&stft/stft_tf.signal.stft/frame/add:z:0*
T0*
_output_shapes
: 2(
&stft/stft_tf.signal.stft/frame/Maximum
(stft/stft_tf.signal.stft/frame/gcd/ConstConst*
_output_shapes
: *
dtype0*
value
B :2*
(stft/stft_tf.signal.stft/frame/gcd/Const
+stft/stft_tf.signal.stft/frame/floordiv_1/yConst*
_output_shapes
: *
dtype0*
value
B :2-
+stft/stft_tf.signal.stft/frame/floordiv_1/yщ
)stft/stft_tf.signal.stft/frame/floordiv_1FloorDiv.stft/stft_tf.signal.stft/frame_length:output:04stft/stft_tf.signal.stft/frame/floordiv_1/y:output:0*
T0*
_output_shapes
: 2+
)stft/stft_tf.signal.stft/frame/floordiv_1
+stft/stft_tf.signal.stft/frame/floordiv_2/yConst*
_output_shapes
: *
dtype0*
value
B :2-
+stft/stft_tf.signal.stft/frame/floordiv_2/yч
)stft/stft_tf.signal.stft/frame/floordiv_2FloorDiv,stft/stft_tf.signal.stft/frame_step:output:04stft/stft_tf.signal.stft/frame/floordiv_2/y:output:0*
T0*
_output_shapes
: 2+
)stft/stft_tf.signal.stft/frame/floordiv_2
+stft/stft_tf.signal.stft/frame/floordiv_3/yConst*
_output_shapes
: *
dtype0*
value
B :2-
+stft/stft_tf.signal.stft/frame/floordiv_3/yъ
)stft/stft_tf.signal.stft/frame/floordiv_3FloorDiv/stft/stft_tf.signal.stft/frame/Reshape:output:04stft/stft_tf.signal.stft/frame/floordiv_3/y:output:0*
T0*
_output_shapes
: 2+
)stft/stft_tf.signal.stft/frame/floordiv_3
$stft/stft_tf.signal.stft/frame/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2&
$stft/stft_tf.signal.stft/frame/mul/yЮ
"stft/stft_tf.signal.stft/frame/mulMul-stft/stft_tf.signal.stft/frame/floordiv_3:z:0-stft/stft_tf.signal.stft/frame/mul/y:output:0*
T0*
_output_shapes
: 2$
"stft/stft_tf.signal.stft/frame/mulО
.stft/stft_tf.signal.stft/frame/concat/values_1Pack&stft/stft_tf.signal.stft/frame/mul:z:0*
N*
T0*
_output_shapes
:20
.stft/stft_tf.signal.stft/frame/concat/values_1
*stft/stft_tf.signal.stft/frame/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2,
*stft/stft_tf.signal.stft/frame/concat/axisд
%stft/stft_tf.signal.stft/frame/concatConcatV2-stft/stft_tf.signal.stft/frame/split:output:07stft/stft_tf.signal.stft/frame/concat/values_1:output:0-stft/stft_tf.signal.stft/frame/split:output:23stft/stft_tf.signal.stft/frame/concat/axis:output:0*
N*
T0*
_output_shapes
:2'
%stft/stft_tf.signal.stft/frame/concatЋ
2stft/stft_tf.signal.stft/frame/concat_1/values_1/1Const*
_output_shapes
: *
dtype0*
value
B :24
2stft/stft_tf.signal.stft/frame/concat_1/values_1/1
0stft/stft_tf.signal.stft/frame/concat_1/values_1Pack-stft/stft_tf.signal.stft/frame/floordiv_3:z:0;stft/stft_tf.signal.stft/frame/concat_1/values_1/1:output:0*
N*
T0*
_output_shapes
:22
0stft/stft_tf.signal.stft/frame/concat_1/values_1
,stft/stft_tf.signal.stft/frame/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,stft/stft_tf.signal.stft/frame/concat_1/axisм
'stft/stft_tf.signal.stft/frame/concat_1ConcatV2-stft/stft_tf.signal.stft/frame/split:output:09stft/stft_tf.signal.stft/frame/concat_1/values_1:output:0-stft/stft_tf.signal.stft/frame/split:output:25stft/stft_tf.signal.stft/frame/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2)
'stft/stft_tf.signal.stft/frame/concat_1 
)stft/stft_tf.signal.stft/frame/zeros_likeConst*
_output_shapes
:*
dtype0*
valueB: 2+
)stft/stft_tf.signal.stft/frame/zeros_likeЊ
.stft/stft_tf.signal.stft/frame/ones_like/ShapeConst*
_output_shapes
:*
dtype0*
valueB:20
.stft/stft_tf.signal.stft/frame/ones_like/ShapeЂ
.stft/stft_tf.signal.stft/frame/ones_like/ConstConst*
_output_shapes
: *
dtype0*
value	B :20
.stft/stft_tf.signal.stft/frame/ones_like/Constѓ
(stft/stft_tf.signal.stft/frame/ones_likeFill7stft/stft_tf.signal.stft/frame/ones_like/Shape:output:07stft/stft_tf.signal.stft/frame/ones_like/Const:output:0*
T0*
_output_shapes
:2*
(stft/stft_tf.signal.stft/frame/ones_likeъ
+stft/stft_tf.signal.stft/frame/StridedSliceStridedSlicestft/transpose:y:02stft/stft_tf.signal.stft/frame/zeros_like:output:0.stft/stft_tf.signal.stft/frame/concat:output:01stft/stft_tf.signal.stft/frame/ones_like:output:0*
Index0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ2-
+stft/stft_tf.signal.stft/frame/StridedSlice
(stft/stft_tf.signal.stft/frame/Reshape_1Reshape4stft/stft_tf.signal.stft/frame/StridedSlice:output:00stft/stft_tf.signal.stft/frame/concat_1:output:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2*
(stft/stft_tf.signal.stft/frame/Reshape_1
,stft/stft_tf.signal.stft/frame/range_1/startConst*
_output_shapes
: *
dtype0*
value	B : 2.
,stft/stft_tf.signal.stft/frame/range_1/start
,stft/stft_tf.signal.stft/frame/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :2.
,stft/stft_tf.signal.stft/frame/range_1/delta
&stft/stft_tf.signal.stft/frame/range_1Range5stft/stft_tf.signal.stft/frame/range_1/start:output:0*stft/stft_tf.signal.stft/frame/Maximum:z:05stft/stft_tf.signal.stft/frame/range_1/delta:output:0*#
_output_shapes
:џџџџџџџџџ2(
&stft/stft_tf.signal.stft/frame/range_1с
$stft/stft_tf.signal.stft/frame/mul_1Mul/stft/stft_tf.signal.stft/frame/range_1:output:0-stft/stft_tf.signal.stft/frame/floordiv_2:z:0*
T0*#
_output_shapes
:џџџџџџџџџ2&
$stft/stft_tf.signal.stft/frame/mul_1І
0stft/stft_tf.signal.stft/frame/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :22
0stft/stft_tf.signal.stft/frame/Reshape_2/shape/1§
.stft/stft_tf.signal.stft/frame/Reshape_2/shapePack*stft/stft_tf.signal.stft/frame/Maximum:z:09stft/stft_tf.signal.stft/frame/Reshape_2/shape/1:output:0*
N*
T0*
_output_shapes
:20
.stft/stft_tf.signal.stft/frame/Reshape_2/shapeє
(stft/stft_tf.signal.stft/frame/Reshape_2Reshape(stft/stft_tf.signal.stft/frame/mul_1:z:07stft/stft_tf.signal.stft/frame/Reshape_2/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2*
(stft/stft_tf.signal.stft/frame/Reshape_2
,stft/stft_tf.signal.stft/frame/range_2/startConst*
_output_shapes
: *
dtype0*
value	B : 2.
,stft/stft_tf.signal.stft/frame/range_2/start
,stft/stft_tf.signal.stft/frame/range_2/deltaConst*
_output_shapes
: *
dtype0*
value	B :2.
,stft/stft_tf.signal.stft/frame/range_2/delta
&stft/stft_tf.signal.stft/frame/range_2Range5stft/stft_tf.signal.stft/frame/range_2/start:output:0-stft/stft_tf.signal.stft/frame/floordiv_1:z:05stft/stft_tf.signal.stft/frame/range_2/delta:output:0*
_output_shapes
:2(
&stft/stft_tf.signal.stft/frame/range_2І
0stft/stft_tf.signal.stft/frame/Reshape_3/shape/0Const*
_output_shapes
: *
dtype0*
value	B :22
0stft/stft_tf.signal.stft/frame/Reshape_3/shape/0
.stft/stft_tf.signal.stft/frame/Reshape_3/shapePack9stft/stft_tf.signal.stft/frame/Reshape_3/shape/0:output:0-stft/stft_tf.signal.stft/frame/floordiv_1:z:0*
N*
T0*
_output_shapes
:20
.stft/stft_tf.signal.stft/frame/Reshape_3/shapeђ
(stft/stft_tf.signal.stft/frame/Reshape_3Reshape/stft/stft_tf.signal.stft/frame/range_2:output:07stft/stft_tf.signal.stft/frame/Reshape_3/shape:output:0*
T0*
_output_shapes

:2*
(stft/stft_tf.signal.stft/frame/Reshape_3э
$stft/stft_tf.signal.stft/frame/add_1AddV21stft/stft_tf.signal.stft/frame/Reshape_2:output:01stft/stft_tf.signal.stft/frame/Reshape_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2&
$stft/stft_tf.signal.stft/frame/add_1ц
'stft/stft_tf.signal.stft/frame/GatherV2GatherV21stft/stft_tf.signal.stft/frame/Reshape_1:output:0(stft/stft_tf.signal.stft/frame/add_1:z:05stft/stft_tf.signal.stft/frame/strided_slice:output:0*
Taxis0*
Tindices0*
Tparams0*F
_output_shapes4
2:0џџџџџџџџџџџџџџџџџџџџџџџџџџџ2)
'stft/stft_tf.signal.stft/frame/GatherV2і
0stft/stft_tf.signal.stft/frame/concat_2/values_1Pack*stft/stft_tf.signal.stft/frame/Maximum:z:0.stft/stft_tf.signal.stft/frame_length:output:0*
N*
T0*
_output_shapes
:22
0stft/stft_tf.signal.stft/frame/concat_2/values_1
,stft/stft_tf.signal.stft/frame/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,stft/stft_tf.signal.stft/frame/concat_2/axisм
'stft/stft_tf.signal.stft/frame/concat_2ConcatV2-stft/stft_tf.signal.stft/frame/split:output:09stft/stft_tf.signal.stft/frame/concat_2/values_1:output:0-stft/stft_tf.signal.stft/frame/split:output:25stft/stft_tf.signal.stft/frame/concat_2/axis:output:0*
N*
T0*
_output_shapes
:2)
'stft/stft_tf.signal.stft/frame/concat_2џ
(stft/stft_tf.signal.stft/frame/Reshape_4Reshape0stft/stft_tf.signal.stft/frame/GatherV2:output:00stft/stft_tf.signal.stft/frame/concat_2:output:0*
T0*1
_output_shapes
:џџџџџџџџџЗ2*
(stft/stft_tf.signal.stft/frame/Reshape_4 
-stft/stft_tf.signal.stft/hann_window/periodicConst*
_output_shapes
: *
dtype0
*
value	B
 Z2/
-stft/stft_tf.signal.stft/hann_window/periodicЦ
)stft/stft_tf.signal.stft/hann_window/CastCast6stft/stft_tf.signal.stft/hann_window/periodic:output:0*

DstT0*

SrcT0
*
_output_shapes
: 2+
)stft/stft_tf.signal.stft/hann_window/CastЄ
/stft/stft_tf.signal.stft/hann_window/FloorMod/yConst*
_output_shapes
: *
dtype0*
value	B :21
/stft/stft_tf.signal.stft/hann_window/FloorMod/yѕ
-stft/stft_tf.signal.stft/hann_window/FloorModFloorMod.stft/stft_tf.signal.stft/frame_length:output:08stft/stft_tf.signal.stft/hann_window/FloorMod/y:output:0*
T0*
_output_shapes
: 2/
-stft/stft_tf.signal.stft/hann_window/FloorMod
*stft/stft_tf.signal.stft/hann_window/sub/xConst*
_output_shapes
: *
dtype0*
value	B :2,
*stft/stft_tf.signal.stft/hann_window/sub/xф
(stft/stft_tf.signal.stft/hann_window/subSub3stft/stft_tf.signal.stft/hann_window/sub/x:output:01stft/stft_tf.signal.stft/hann_window/FloorMod:z:0*
T0*
_output_shapes
: 2*
(stft/stft_tf.signal.stft/hann_window/subй
(stft/stft_tf.signal.stft/hann_window/mulMul-stft/stft_tf.signal.stft/hann_window/Cast:y:0,stft/stft_tf.signal.stft/hann_window/sub:z:0*
T0*
_output_shapes
: 2*
(stft/stft_tf.signal.stft/hann_window/mulм
(stft/stft_tf.signal.stft/hann_window/addAddV2.stft/stft_tf.signal.stft/frame_length:output:0,stft/stft_tf.signal.stft/hann_window/mul:z:0*
T0*
_output_shapes
: 2*
(stft/stft_tf.signal.stft/hann_window/add
,stft/stft_tf.signal.stft/hann_window/sub_1/yConst*
_output_shapes
: *
dtype0*
value	B :2.
,stft/stft_tf.signal.stft/hann_window/sub_1/yх
*stft/stft_tf.signal.stft/hann_window/sub_1Sub,stft/stft_tf.signal.stft/hann_window/add:z:05stft/stft_tf.signal.stft/hann_window/sub_1/y:output:0*
T0*
_output_shapes
: 2,
*stft/stft_tf.signal.stft/hann_window/sub_1Т
+stft/stft_tf.signal.stft/hann_window/Cast_1Cast.stft/stft_tf.signal.stft/hann_window/sub_1:z:0*

DstT0*

SrcT0*
_output_shapes
: 2-
+stft/stft_tf.signal.stft/hann_window/Cast_1І
0stft/stft_tf.signal.stft/hann_window/range/startConst*
_output_shapes
: *
dtype0*
value	B : 22
0stft/stft_tf.signal.stft/hann_window/range/startІ
0stft/stft_tf.signal.stft/hann_window/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :22
0stft/stft_tf.signal.stft/hann_window/range/deltaЄ
*stft/stft_tf.signal.stft/hann_window/rangeRange9stft/stft_tf.signal.stft/hann_window/range/start:output:0.stft/stft_tf.signal.stft/frame_length:output:09stft/stft_tf.signal.stft/hann_window/range/delta:output:0*
_output_shapes	
:2,
*stft/stft_tf.signal.stft/hann_window/rangeЬ
+stft/stft_tf.signal.stft/hann_window/Cast_2Cast3stft/stft_tf.signal.stft/hann_window/range:output:0*

DstT0*

SrcT0*
_output_shapes	
:2-
+stft/stft_tf.signal.stft/hann_window/Cast_2
*stft/stft_tf.signal.stft/hann_window/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *лЩ@2,
*stft/stft_tf.signal.stft/hann_window/Constы
*stft/stft_tf.signal.stft/hann_window/mul_1Mul3stft/stft_tf.signal.stft/hann_window/Const:output:0/stft/stft_tf.signal.stft/hann_window/Cast_2:y:0*
T0*
_output_shapes	
:2,
*stft/stft_tf.signal.stft/hann_window/mul_1ю
,stft/stft_tf.signal.stft/hann_window/truedivRealDiv.stft/stft_tf.signal.stft/hann_window/mul_1:z:0/stft/stft_tf.signal.stft/hann_window/Cast_1:y:0*
T0*
_output_shapes	
:2.
,stft/stft_tf.signal.stft/hann_window/truedivГ
(stft/stft_tf.signal.stft/hann_window/CosCos0stft/stft_tf.signal.stft/hann_window/truediv:z:0*
T0*
_output_shapes	
:2*
(stft/stft_tf.signal.stft/hann_window/CosЁ
,stft/stft_tf.signal.stft/hann_window/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2.
,stft/stft_tf.signal.stft/hann_window/mul_2/xъ
*stft/stft_tf.signal.stft/hann_window/mul_2Mul5stft/stft_tf.signal.stft/hann_window/mul_2/x:output:0,stft/stft_tf.signal.stft/hann_window/Cos:y:0*
T0*
_output_shapes	
:2,
*stft/stft_tf.signal.stft/hann_window/mul_2Ё
,stft/stft_tf.signal.stft/hann_window/sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2.
,stft/stft_tf.signal.stft/hann_window/sub_2/xь
*stft/stft_tf.signal.stft/hann_window/sub_2Sub5stft/stft_tf.signal.stft/hann_window/sub_2/x:output:0.stft/stft_tf.signal.stft/hann_window/mul_2:z:0*
T0*
_output_shapes	
:2,
*stft/stft_tf.signal.stft/hann_window/sub_2т
stft/stft_tf.signal.stft/mulMul1stft/stft_tf.signal.stft/frame/Reshape_4:output:0.stft/stft_tf.signal.stft/hann_window/sub_2:z:0*
T0*1
_output_shapes
:џџџџџџџџџЗ2
stft/stft_tf.signal.stft/mulА
$stft/stft_tf.signal.stft/rfft/packedPack,stft/stft_tf.signal.stft/fft_length:output:0*
N*
T0*
_output_shapes
:2&
$stft/stft_tf.signal.stft/rfft/packed
(stft/stft_tf.signal.stft/rfft/fft_lengthConst*
_output_shapes
:*
dtype0*
valueB:2*
(stft/stft_tf.signal.stft/rfft/fft_lengthЮ
stft/stft_tf.signal.stft/rfftRFFT stft/stft_tf.signal.stft/mul:z:01stft/stft_tf.signal.stft/rfft/fft_length:output:0*1
_output_shapes
:џџџџџџџџџЗ2
stft/stft_tf.signal.stft/rfft
stft/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             2
stft/transpose_1/permЕ
stft/transpose_1	Transpose&stft/stft_tf.signal.stft/rfft:output:0stft/transpose_1/perm:output:0*
T0*1
_output_shapes
:џџџџџџџџџЗ2
stft/transpose_1u
magnitude/Abs
ComplexAbsstft/transpose_1:y:0*1
_output_shapes
:џџџџџџџџџЗ2
magnitude/Abs
apply_filterbank/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2!
apply_filterbank/Tensordot/axes
apply_filterbank/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          2!
apply_filterbank/Tensordot/free
 apply_filterbank/Tensordot/ShapeShapemagnitude/Abs:y:0*
T0*
_output_shapes
:2"
 apply_filterbank/Tensordot/Shape
(apply_filterbank/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2*
(apply_filterbank/Tensordot/GatherV2/axisІ
#apply_filterbank/Tensordot/GatherV2GatherV2)apply_filterbank/Tensordot/Shape:output:0(apply_filterbank/Tensordot/free:output:01apply_filterbank/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2%
#apply_filterbank/Tensordot/GatherV2
*apply_filterbank/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2,
*apply_filterbank/Tensordot/GatherV2_1/axisЌ
%apply_filterbank/Tensordot/GatherV2_1GatherV2)apply_filterbank/Tensordot/Shape:output:0(apply_filterbank/Tensordot/axes:output:03apply_filterbank/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2'
%apply_filterbank/Tensordot/GatherV2_1
 apply_filterbank/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 apply_filterbank/Tensordot/ConstФ
apply_filterbank/Tensordot/ProdProd,apply_filterbank/Tensordot/GatherV2:output:0)apply_filterbank/Tensordot/Const:output:0*
T0*
_output_shapes
: 2!
apply_filterbank/Tensordot/Prod
"apply_filterbank/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2$
"apply_filterbank/Tensordot/Const_1Ь
!apply_filterbank/Tensordot/Prod_1Prod.apply_filterbank/Tensordot/GatherV2_1:output:0+apply_filterbank/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2#
!apply_filterbank/Tensordot/Prod_1
&apply_filterbank/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2(
&apply_filterbank/Tensordot/concat/axis
!apply_filterbank/Tensordot/concatConcatV2(apply_filterbank/Tensordot/free:output:0(apply_filterbank/Tensordot/axes:output:0/apply_filterbank/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2#
!apply_filterbank/Tensordot/concatа
 apply_filterbank/Tensordot/stackPack(apply_filterbank/Tensordot/Prod:output:0*apply_filterbank/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2"
 apply_filterbank/Tensordot/stackд
$apply_filterbank/Tensordot/transpose	Transposemagnitude/Abs:y:0*apply_filterbank/Tensordot/concat:output:0*
T0*1
_output_shapes
:џџџџџџџџџЗ2&
$apply_filterbank/Tensordot/transposeу
"apply_filterbank/Tensordot/ReshapeReshape(apply_filterbank/Tensordot/transpose:y:0)apply_filterbank/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2$
"apply_filterbank/Tensordot/ReshapeЮ
!apply_filterbank/Tensordot/MatMulMatMul+apply_filterbank/Tensordot/Reshape:output:0apply_filterbank_tensordot_b*
T0*(
_output_shapes
:џџџџџџџџџ2#
!apply_filterbank/Tensordot/MatMul
"apply_filterbank/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"apply_filterbank/Tensordot/Const_2
(apply_filterbank/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2*
(apply_filterbank/Tensordot/concat_1/axis
#apply_filterbank/Tensordot/concat_1ConcatV2,apply_filterbank/Tensordot/GatherV2:output:0+apply_filterbank/Tensordot/Const_2:output:01apply_filterbank/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2%
#apply_filterbank/Tensordot/concat_1к
apply_filterbank/TensordotReshape+apply_filterbank/Tensordot/MatMul:product:0,apply_filterbank/Tensordot/concat_1:output:0*
T0*1
_output_shapes
:џџџџџџџџџЗ2
apply_filterbank/Tensordot
apply_filterbank/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             2!
apply_filterbank/transpose/permа
apply_filterbank/transpose	Transpose#apply_filterbank/Tensordot:output:0(apply_filterbank/transpose/perm:output:0*
T0*1
_output_shapes
:џџџџџџџџџЗ2
apply_filterbank/transpose|
IdentityIdentityapply_filterbank/transpose:y:0*
T0*1
_output_shapes
:џџџџџџџџџЗ2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:џџџџџџџџџт	:
:U Q
-
_output_shapes
:џџџџџџџџџт	
 
_user_specified_nameinputs:&"
 
_output_shapes
:


C
'__inference_dropout_layer_call_fn_23021

inputs
identityР
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_209092
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*&
_input_shapes
:џџџџџџџџџ@:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs

;
$__inference_stft_layer_call_fn_23202
x
identityТ
PartitionedCallPartitionedCallx*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:џџџџџџџџџЗ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_stft_layer_call_and_return_conditional_losses_198772
PartitionedCallv
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:џџџџџџџџџЗ2

Identity"
identityIdentity:output:0*,
_input_shapes
:џџџџџџџџџт	:P L
-
_output_shapes
:џџџџџџџџџт	

_user_specified_namex
њ
}
(__inference_conv2d_2_layer_call_fn_22835

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallћ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ"  *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_conv2d_2_layer_call_and_return_conditional_losses_207612
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:џџџџџџџџџ"  2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџ" ::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ" 
 
_user_specified_nameinputs
я
V
?__inference_stft_layer_call_and_return_conditional_losses_23197
x
identityu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permw
	transpose	Transposextranspose/perm:output:0*
T0*-
_output_shapes
:џџџџџџџџџт	2
	transpose
 stft_tf.signal.stft/frame_lengthConst*
_output_shapes
: *
dtype0*
value
B :2"
 stft_tf.signal.stft/frame_length
stft_tf.signal.stft/frame_stepConst*
_output_shapes
: *
dtype0*
value
B :2 
stft_tf.signal.stft/frame_step
stft_tf.signal.stft/fft_lengthConst*
_output_shapes
: *
dtype0*
value
B :2 
stft_tf.signal.stft/fft_length
stft_tf.signal.stft/frame/axisConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2 
stft_tf.signal.stft/frame/axis
stft_tf.signal.stft/frame/ShapeShapetranspose:y:0*
T0*
_output_shapes
:2!
stft_tf.signal.stft/frame/Shape
stft_tf.signal.stft/frame/RankConst*
_output_shapes
: *
dtype0*
value	B :2 
stft_tf.signal.stft/frame/Rank
%stft_tf.signal.stft/frame/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2'
%stft_tf.signal.stft/frame/range/start
%stft_tf.signal.stft/frame/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2'
%stft_tf.signal.stft/frame/range/delta№
stft_tf.signal.stft/frame/rangeRange.stft_tf.signal.stft/frame/range/start:output:0'stft_tf.signal.stft/frame/Rank:output:0.stft_tf.signal.stft/frame/range/delta:output:0*
_output_shapes
:2!
stft_tf.signal.stft/frame/rangeБ
-stft_tf.signal.stft/frame/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2/
-stft_tf.signal.stft/frame/strided_slice/stackЌ
/stft_tf.signal.stft/frame/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 21
/stft_tf.signal.stft/frame/strided_slice/stack_1Ќ
/stft_tf.signal.stft/frame/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/stft_tf.signal.stft/frame/strided_slice/stack_2ў
'stft_tf.signal.stft/frame/strided_sliceStridedSlice(stft_tf.signal.stft/frame/range:output:06stft_tf.signal.stft/frame/strided_slice/stack:output:08stft_tf.signal.stft/frame/strided_slice/stack_1:output:08stft_tf.signal.stft/frame/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2)
'stft_tf.signal.stft/frame/strided_slice
stft_tf.signal.stft/frame/sub/yConst*
_output_shapes
: *
dtype0*
value	B :2!
stft_tf.signal.stft/frame/sub/yЙ
stft_tf.signal.stft/frame/subSub'stft_tf.signal.stft/frame/Rank:output:0(stft_tf.signal.stft/frame/sub/y:output:0*
T0*
_output_shapes
: 2
stft_tf.signal.stft/frame/subП
stft_tf.signal.stft/frame/sub_1Sub!stft_tf.signal.stft/frame/sub:z:00stft_tf.signal.stft/frame/strided_slice:output:0*
T0*
_output_shapes
: 2!
stft_tf.signal.stft/frame/sub_1
"stft_tf.signal.stft/frame/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2$
"stft_tf.signal.stft/frame/packed/1ў
 stft_tf.signal.stft/frame/packedPack0stft_tf.signal.stft/frame/strided_slice:output:0+stft_tf.signal.stft/frame/packed/1:output:0#stft_tf.signal.stft/frame/sub_1:z:0*
N*
T0*
_output_shapes
:2"
 stft_tf.signal.stft/frame/packed
)stft_tf.signal.stft/frame/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2+
)stft_tf.signal.stft/frame/split/split_dimЁ
stft_tf.signal.stft/frame/splitSplitV(stft_tf.signal.stft/frame/Shape:output:0)stft_tf.signal.stft/frame/packed:output:02stft_tf.signal.stft/frame/split/split_dim:output:0*
T0*

Tlen0*$
_output_shapes
::: *
	num_split2!
stft_tf.signal.stft/frame/split
'stft_tf.signal.stft/frame/Reshape/shapeConst*
_output_shapes
: *
dtype0*
valueB 2)
'stft_tf.signal.stft/frame/Reshape/shape
)stft_tf.signal.stft/frame/Reshape/shape_1Const*
_output_shapes
: *
dtype0*
valueB 2+
)stft_tf.signal.stft/frame/Reshape/shape_1а
!stft_tf.signal.stft/frame/ReshapeReshape(stft_tf.signal.stft/frame/split:output:12stft_tf.signal.stft/frame/Reshape/shape_1:output:0*
T0*
_output_shapes
: 2#
!stft_tf.signal.stft/frame/Reshape
stft_tf.signal.stft/frame/SizeConst*
_output_shapes
: *
dtype0*
value	B :2 
stft_tf.signal.stft/frame/Size
 stft_tf.signal.stft/frame/Size_1Const*
_output_shapes
: *
dtype0*
value	B : 2"
 stft_tf.signal.stft/frame/Size_1С
stft_tf.signal.stft/frame/sub_2Sub*stft_tf.signal.stft/frame/Reshape:output:0)stft_tf.signal.stft/frame_length:output:0*
T0*
_output_shapes
: 2!
stft_tf.signal.stft/frame/sub_2У
"stft_tf.signal.stft/frame/floordivFloorDiv#stft_tf.signal.stft/frame/sub_2:z:0'stft_tf.signal.stft/frame_step:output:0*
T0*
_output_shapes
: 2$
"stft_tf.signal.stft/frame/floordiv
stft_tf.signal.stft/frame/add/xConst*
_output_shapes
: *
dtype0*
value	B :2!
stft_tf.signal.stft/frame/add/xК
stft_tf.signal.stft/frame/addAddV2(stft_tf.signal.stft/frame/add/x:output:0&stft_tf.signal.stft/frame/floordiv:z:0*
T0*
_output_shapes
: 2
stft_tf.signal.stft/frame/add
#stft_tf.signal.stft/frame/Maximum/xConst*
_output_shapes
: *
dtype0*
value	B : 2%
#stft_tf.signal.stft/frame/Maximum/xУ
!stft_tf.signal.stft/frame/MaximumMaximum,stft_tf.signal.stft/frame/Maximum/x:output:0!stft_tf.signal.stft/frame/add:z:0*
T0*
_output_shapes
: 2#
!stft_tf.signal.stft/frame/Maximum
#stft_tf.signal.stft/frame/gcd/ConstConst*
_output_shapes
: *
dtype0*
value
B :2%
#stft_tf.signal.stft/frame/gcd/Const
&stft_tf.signal.stft/frame/floordiv_1/yConst*
_output_shapes
: *
dtype0*
value
B :2(
&stft_tf.signal.stft/frame/floordiv_1/yе
$stft_tf.signal.stft/frame/floordiv_1FloorDiv)stft_tf.signal.stft/frame_length:output:0/stft_tf.signal.stft/frame/floordiv_1/y:output:0*
T0*
_output_shapes
: 2&
$stft_tf.signal.stft/frame/floordiv_1
&stft_tf.signal.stft/frame/floordiv_2/yConst*
_output_shapes
: *
dtype0*
value
B :2(
&stft_tf.signal.stft/frame/floordiv_2/yг
$stft_tf.signal.stft/frame/floordiv_2FloorDiv'stft_tf.signal.stft/frame_step:output:0/stft_tf.signal.stft/frame/floordiv_2/y:output:0*
T0*
_output_shapes
: 2&
$stft_tf.signal.stft/frame/floordiv_2
&stft_tf.signal.stft/frame/floordiv_3/yConst*
_output_shapes
: *
dtype0*
value
B :2(
&stft_tf.signal.stft/frame/floordiv_3/yж
$stft_tf.signal.stft/frame/floordiv_3FloorDiv*stft_tf.signal.stft/frame/Reshape:output:0/stft_tf.signal.stft/frame/floordiv_3/y:output:0*
T0*
_output_shapes
: 2&
$stft_tf.signal.stft/frame/floordiv_3
stft_tf.signal.stft/frame/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2!
stft_tf.signal.stft/frame/mul/yК
stft_tf.signal.stft/frame/mulMul(stft_tf.signal.stft/frame/floordiv_3:z:0(stft_tf.signal.stft/frame/mul/y:output:0*
T0*
_output_shapes
: 2
stft_tf.signal.stft/frame/mulЏ
)stft_tf.signal.stft/frame/concat/values_1Pack!stft_tf.signal.stft/frame/mul:z:0*
N*
T0*
_output_shapes
:2+
)stft_tf.signal.stft/frame/concat/values_1
%stft_tf.signal.stft/frame/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2'
%stft_tf.signal.stft/frame/concat/axisЖ
 stft_tf.signal.stft/frame/concatConcatV2(stft_tf.signal.stft/frame/split:output:02stft_tf.signal.stft/frame/concat/values_1:output:0(stft_tf.signal.stft/frame/split:output:2.stft_tf.signal.stft/frame/concat/axis:output:0*
N*
T0*
_output_shapes
:2"
 stft_tf.signal.stft/frame/concatЁ
-stft_tf.signal.stft/frame/concat_1/values_1/1Const*
_output_shapes
: *
dtype0*
value
B :2/
-stft_tf.signal.stft/frame/concat_1/values_1/1ђ
+stft_tf.signal.stft/frame/concat_1/values_1Pack(stft_tf.signal.stft/frame/floordiv_3:z:06stft_tf.signal.stft/frame/concat_1/values_1/1:output:0*
N*
T0*
_output_shapes
:2-
+stft_tf.signal.stft/frame/concat_1/values_1
'stft_tf.signal.stft/frame/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2)
'stft_tf.signal.stft/frame/concat_1/axisО
"stft_tf.signal.stft/frame/concat_1ConcatV2(stft_tf.signal.stft/frame/split:output:04stft_tf.signal.stft/frame/concat_1/values_1:output:0(stft_tf.signal.stft/frame/split:output:20stft_tf.signal.stft/frame/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2$
"stft_tf.signal.stft/frame/concat_1
$stft_tf.signal.stft/frame/zeros_likeConst*
_output_shapes
:*
dtype0*
valueB: 2&
$stft_tf.signal.stft/frame/zeros_like 
)stft_tf.signal.stft/frame/ones_like/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2+
)stft_tf.signal.stft/frame/ones_like/Shape
)stft_tf.signal.stft/frame/ones_like/ConstConst*
_output_shapes
: *
dtype0*
value	B :2+
)stft_tf.signal.stft/frame/ones_like/Constп
#stft_tf.signal.stft/frame/ones_likeFill2stft_tf.signal.stft/frame/ones_like/Shape:output:02stft_tf.signal.stft/frame/ones_like/Const:output:0*
T0*
_output_shapes
:2%
#stft_tf.signal.stft/frame/ones_likeЬ
&stft_tf.signal.stft/frame/StridedSliceStridedSlicetranspose:y:0-stft_tf.signal.stft/frame/zeros_like:output:0)stft_tf.signal.stft/frame/concat:output:0,stft_tf.signal.stft/frame/ones_like:output:0*
Index0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ2(
&stft_tf.signal.stft/frame/StridedSlice
#stft_tf.signal.stft/frame/Reshape_1Reshape/stft_tf.signal.stft/frame/StridedSlice:output:0+stft_tf.signal.stft/frame/concat_1:output:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2%
#stft_tf.signal.stft/frame/Reshape_1
'stft_tf.signal.stft/frame/range_1/startConst*
_output_shapes
: *
dtype0*
value	B : 2)
'stft_tf.signal.stft/frame/range_1/start
'stft_tf.signal.stft/frame/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :2)
'stft_tf.signal.stft/frame/range_1/deltaџ
!stft_tf.signal.stft/frame/range_1Range0stft_tf.signal.stft/frame/range_1/start:output:0%stft_tf.signal.stft/frame/Maximum:z:00stft_tf.signal.stft/frame/range_1/delta:output:0*#
_output_shapes
:џџџџџџџџџ2#
!stft_tf.signal.stft/frame/range_1Э
stft_tf.signal.stft/frame/mul_1Mul*stft_tf.signal.stft/frame/range_1:output:0(stft_tf.signal.stft/frame/floordiv_2:z:0*
T0*#
_output_shapes
:џџџџџџџџџ2!
stft_tf.signal.stft/frame/mul_1
+stft_tf.signal.stft/frame/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2-
+stft_tf.signal.stft/frame/Reshape_2/shape/1щ
)stft_tf.signal.stft/frame/Reshape_2/shapePack%stft_tf.signal.stft/frame/Maximum:z:04stft_tf.signal.stft/frame/Reshape_2/shape/1:output:0*
N*
T0*
_output_shapes
:2+
)stft_tf.signal.stft/frame/Reshape_2/shapeр
#stft_tf.signal.stft/frame/Reshape_2Reshape#stft_tf.signal.stft/frame/mul_1:z:02stft_tf.signal.stft/frame/Reshape_2/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2%
#stft_tf.signal.stft/frame/Reshape_2
'stft_tf.signal.stft/frame/range_2/startConst*
_output_shapes
: *
dtype0*
value	B : 2)
'stft_tf.signal.stft/frame/range_2/start
'stft_tf.signal.stft/frame/range_2/deltaConst*
_output_shapes
: *
dtype0*
value	B :2)
'stft_tf.signal.stft/frame/range_2/deltaљ
!stft_tf.signal.stft/frame/range_2Range0stft_tf.signal.stft/frame/range_2/start:output:0(stft_tf.signal.stft/frame/floordiv_1:z:00stft_tf.signal.stft/frame/range_2/delta:output:0*
_output_shapes
:2#
!stft_tf.signal.stft/frame/range_2
+stft_tf.signal.stft/frame/Reshape_3/shape/0Const*
_output_shapes
: *
dtype0*
value	B :2-
+stft_tf.signal.stft/frame/Reshape_3/shape/0ь
)stft_tf.signal.stft/frame/Reshape_3/shapePack4stft_tf.signal.stft/frame/Reshape_3/shape/0:output:0(stft_tf.signal.stft/frame/floordiv_1:z:0*
N*
T0*
_output_shapes
:2+
)stft_tf.signal.stft/frame/Reshape_3/shapeо
#stft_tf.signal.stft/frame/Reshape_3Reshape*stft_tf.signal.stft/frame/range_2:output:02stft_tf.signal.stft/frame/Reshape_3/shape:output:0*
T0*
_output_shapes

:2%
#stft_tf.signal.stft/frame/Reshape_3й
stft_tf.signal.stft/frame/add_1AddV2,stft_tf.signal.stft/frame/Reshape_2:output:0,stft_tf.signal.stft/frame/Reshape_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2!
stft_tf.signal.stft/frame/add_1Э
"stft_tf.signal.stft/frame/GatherV2GatherV2,stft_tf.signal.stft/frame/Reshape_1:output:0#stft_tf.signal.stft/frame/add_1:z:00stft_tf.signal.stft/frame/strided_slice:output:0*
Taxis0*
Tindices0*
Tparams0*F
_output_shapes4
2:0џџџџџџџџџџџџџџџџџџџџџџџџџџџ2$
"stft_tf.signal.stft/frame/GatherV2т
+stft_tf.signal.stft/frame/concat_2/values_1Pack%stft_tf.signal.stft/frame/Maximum:z:0)stft_tf.signal.stft/frame_length:output:0*
N*
T0*
_output_shapes
:2-
+stft_tf.signal.stft/frame/concat_2/values_1
'stft_tf.signal.stft/frame/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2)
'stft_tf.signal.stft/frame/concat_2/axisО
"stft_tf.signal.stft/frame/concat_2ConcatV2(stft_tf.signal.stft/frame/split:output:04stft_tf.signal.stft/frame/concat_2/values_1:output:0(stft_tf.signal.stft/frame/split:output:20stft_tf.signal.stft/frame/concat_2/axis:output:0*
N*
T0*
_output_shapes
:2$
"stft_tf.signal.stft/frame/concat_2ы
#stft_tf.signal.stft/frame/Reshape_4Reshape+stft_tf.signal.stft/frame/GatherV2:output:0+stft_tf.signal.stft/frame/concat_2:output:0*
T0*1
_output_shapes
:џџџџџџџџџЗ2%
#stft_tf.signal.stft/frame/Reshape_4
(stft_tf.signal.stft/hann_window/periodicConst*
_output_shapes
: *
dtype0
*
value	B
 Z2*
(stft_tf.signal.stft/hann_window/periodicЗ
$stft_tf.signal.stft/hann_window/CastCast1stft_tf.signal.stft/hann_window/periodic:output:0*

DstT0*

SrcT0
*
_output_shapes
: 2&
$stft_tf.signal.stft/hann_window/Cast
*stft_tf.signal.stft/hann_window/FloorMod/yConst*
_output_shapes
: *
dtype0*
value	B :2,
*stft_tf.signal.stft/hann_window/FloorMod/yс
(stft_tf.signal.stft/hann_window/FloorModFloorMod)stft_tf.signal.stft/frame_length:output:03stft_tf.signal.stft/hann_window/FloorMod/y:output:0*
T0*
_output_shapes
: 2*
(stft_tf.signal.stft/hann_window/FloorMod
%stft_tf.signal.stft/hann_window/sub/xConst*
_output_shapes
: *
dtype0*
value	B :2'
%stft_tf.signal.stft/hann_window/sub/xа
#stft_tf.signal.stft/hann_window/subSub.stft_tf.signal.stft/hann_window/sub/x:output:0,stft_tf.signal.stft/hann_window/FloorMod:z:0*
T0*
_output_shapes
: 2%
#stft_tf.signal.stft/hann_window/subХ
#stft_tf.signal.stft/hann_window/mulMul(stft_tf.signal.stft/hann_window/Cast:y:0'stft_tf.signal.stft/hann_window/sub:z:0*
T0*
_output_shapes
: 2%
#stft_tf.signal.stft/hann_window/mulШ
#stft_tf.signal.stft/hann_window/addAddV2)stft_tf.signal.stft/frame_length:output:0'stft_tf.signal.stft/hann_window/mul:z:0*
T0*
_output_shapes
: 2%
#stft_tf.signal.stft/hann_window/add
'stft_tf.signal.stft/hann_window/sub_1/yConst*
_output_shapes
: *
dtype0*
value	B :2)
'stft_tf.signal.stft/hann_window/sub_1/yб
%stft_tf.signal.stft/hann_window/sub_1Sub'stft_tf.signal.stft/hann_window/add:z:00stft_tf.signal.stft/hann_window/sub_1/y:output:0*
T0*
_output_shapes
: 2'
%stft_tf.signal.stft/hann_window/sub_1Г
&stft_tf.signal.stft/hann_window/Cast_1Cast)stft_tf.signal.stft/hann_window/sub_1:z:0*

DstT0*

SrcT0*
_output_shapes
: 2(
&stft_tf.signal.stft/hann_window/Cast_1
+stft_tf.signal.stft/hann_window/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2-
+stft_tf.signal.stft/hann_window/range/start
+stft_tf.signal.stft/hann_window/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2-
+stft_tf.signal.stft/hann_window/range/delta
%stft_tf.signal.stft/hann_window/rangeRange4stft_tf.signal.stft/hann_window/range/start:output:0)stft_tf.signal.stft/frame_length:output:04stft_tf.signal.stft/hann_window/range/delta:output:0*
_output_shapes	
:2'
%stft_tf.signal.stft/hann_window/rangeН
&stft_tf.signal.stft/hann_window/Cast_2Cast.stft_tf.signal.stft/hann_window/range:output:0*

DstT0*

SrcT0*
_output_shapes	
:2(
&stft_tf.signal.stft/hann_window/Cast_2
%stft_tf.signal.stft/hann_window/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *лЩ@2'
%stft_tf.signal.stft/hann_window/Constз
%stft_tf.signal.stft/hann_window/mul_1Mul.stft_tf.signal.stft/hann_window/Const:output:0*stft_tf.signal.stft/hann_window/Cast_2:y:0*
T0*
_output_shapes	
:2'
%stft_tf.signal.stft/hann_window/mul_1к
'stft_tf.signal.stft/hann_window/truedivRealDiv)stft_tf.signal.stft/hann_window/mul_1:z:0*stft_tf.signal.stft/hann_window/Cast_1:y:0*
T0*
_output_shapes	
:2)
'stft_tf.signal.stft/hann_window/truedivЄ
#stft_tf.signal.stft/hann_window/CosCos+stft_tf.signal.stft/hann_window/truediv:z:0*
T0*
_output_shapes	
:2%
#stft_tf.signal.stft/hann_window/Cos
'stft_tf.signal.stft/hann_window/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2)
'stft_tf.signal.stft/hann_window/mul_2/xж
%stft_tf.signal.stft/hann_window/mul_2Mul0stft_tf.signal.stft/hann_window/mul_2/x:output:0'stft_tf.signal.stft/hann_window/Cos:y:0*
T0*
_output_shapes	
:2'
%stft_tf.signal.stft/hann_window/mul_2
'stft_tf.signal.stft/hann_window/sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2)
'stft_tf.signal.stft/hann_window/sub_2/xи
%stft_tf.signal.stft/hann_window/sub_2Sub0stft_tf.signal.stft/hann_window/sub_2/x:output:0)stft_tf.signal.stft/hann_window/mul_2:z:0*
T0*
_output_shapes	
:2'
%stft_tf.signal.stft/hann_window/sub_2Ю
stft_tf.signal.stft/mulMul,stft_tf.signal.stft/frame/Reshape_4:output:0)stft_tf.signal.stft/hann_window/sub_2:z:0*
T0*1
_output_shapes
:џџџџџџџџџЗ2
stft_tf.signal.stft/mulЁ
stft_tf.signal.stft/rfft/packedPack'stft_tf.signal.stft/fft_length:output:0*
N*
T0*
_output_shapes
:2!
stft_tf.signal.stft/rfft/packed
#stft_tf.signal.stft/rfft/fft_lengthConst*
_output_shapes
:*
dtype0*
valueB:2%
#stft_tf.signal.stft/rfft/fft_lengthК
stft_tf.signal.stft/rfftRFFTstft_tf.signal.stft/mul:z:0,stft_tf.signal.stft/rfft/fft_length:output:0*1
_output_shapes
:џџџџџџџџџЗ2
stft_tf.signal.stft/rfft}
transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             2
transpose_1/permЁ
transpose_1	Transpose!stft_tf.signal.stft/rfft:output:0transpose_1/perm:output:0*
T0*1
_output_shapes
:џџџџџџџџџЗ2
transpose_1m
IdentityIdentitytranspose_1:y:0*
T0*1
_output_shapes
:џџџџџџџџџЗ2

Identity"
identityIdentity:output:0*,
_input_shapes
:џџџџџџџџџт	:P L
-
_output_shapes
:џџџџџџџџџт	

_user_specified_namex
г
Ј
5__inference_batch_normalization_1_layer_call_fn_22654

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityЂStatefulPartitionedCall 
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџg@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_205952
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:џџџџџџџџџg@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџg@::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџg@
 
_user_specified_nameinputs
ј
ѓ
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_22789

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ" :::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3к
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:џџџџџџџџџ" 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ" ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:џџџџџџџџџ" 
 
_user_specified_nameinputs
Э

м
C__inference_conv2d_2_layer_call_and_return_conditional_losses_20761

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOpЃ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ"  *
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ"  2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ"  2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:џџџџџџџџџ"  2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџ" ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ" 
 
_user_specified_nameinputs

W
.__inference_melspectrogram_layer_call_fn_22384

inputs
unknown
identityл
PartitionedCallPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:џџџџџџџџџЗ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_melspectrogram_layer_call_and_return_conditional_losses_199602
PartitionedCallv
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:џџџџџџџџџЗ2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:џџџџџџџџџт	:
:U Q
-
_output_shapes
:џџџџџџџџџт	
 
_user_specified_nameinputs:&"
 
_output_shapes
:


c
D__inference_dropout_1_layer_call_and_return_conditional_losses_23053

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeД
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>2
dropout/GreaterEqual/yО
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*&
_input_shapes
:џџџџџџџџџ:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ф
s
K__inference_apply_filterbank_layer_call_and_return_conditional_losses_19926
x
tensordot_b
identityj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesu
Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          2
Tensordot/freeS
Tensordot/ShapeShapex*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axisб
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axisз
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axisА
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack
Tensordot/transpose	TransposexTensordot/concat:output:0*
T0*1
_output_shapes
:џџџџџџџџџЗ2
Tensordot/transpose
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2
Tensordot/Reshape
Tensordot/MatMulMatMulTensordot/Reshape:output:0tensordot_b*
T0*(
_output_shapes
:џџџџџџџџџ2
Tensordot/MatMulq
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axisН
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*1
_output_shapes
:џџџџџџџџџЗ2
	Tensordoty
transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             2
transpose/perm
	transpose	TransposeTensordot:output:0transpose/perm:output:0*
T0*1
_output_shapes
:џџџџџџџџџЗ2
	transposek
IdentityIdentitytranspose:y:0*
T0*1
_output_shapes
:џџџџџџџџџЗ2

Identity"
identityIdentity:output:0*<
_input_shapes+
):џџџџџџџџџЗ:
:T P
1
_output_shapes
:џџџџџџџџџЗ

_user_specified_namex:&"
 
_output_shapes
:

э

I__inference_melspectrogram_layer_call_and_return_conditional_losses_19948

stft_input
apply_filterbank_19944
identityе
stft/PartitionedCallPartitionedCall
stft_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:џџџџџџџџџЗ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_stft_layer_call_and_return_conditional_losses_198772
stft/PartitionedCallї
magnitude/PartitionedCallPartitionedCallstft/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:џџџџџџџџџЗ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_magnitude_layer_call_and_return_conditional_losses_198902
magnitude/PartitionedCallЊ
 apply_filterbank/PartitionedCallPartitionedCall"magnitude/PartitionedCall:output:0apply_filterbank_19944*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:џџџџџџџџџЗ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_apply_filterbank_layer_call_and_return_conditional_losses_199262"
 apply_filterbank/PartitionedCall
IdentityIdentity)apply_filterbank/PartitionedCall:output:0*
T0*1
_output_shapes
:џџџџџџџџџЗ2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:џџџџџџџџџт	:
:Y U
-
_output_shapes
:џџџџџџџџџт	
$
_user_specified_name
stft_input:&"
 
_output_shapes
:

и
|
'__inference_dense_2_layer_call_fn_23088

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallђ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_209902
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

[
.__inference_melspectrogram_layer_call_fn_19965

stft_input
unknown
identityп
PartitionedCallPartitionedCall
stft_inputunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:џџџџџџџџџЗ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_melspectrogram_layer_call_and_return_conditional_losses_199602
PartitionedCallv
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:џџџџџџџџџЗ2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:џџџџџџџџџт	:
:Y U
-
_output_shapes
:џџџџџџџџџт	
$
_user_specified_name
stft_input:&"
 
_output_shapes
:


Ј
5__inference_batch_normalization_2_layer_call_fn_22751

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityЂStatefulPartitionedCallД
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_203062
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Л
[
D__inference_magnitude_layer_call_and_return_conditional_losses_23207
x
identityN
Abs
ComplexAbsx*1
_output_shapes
:џџџџџџџџџЗ2
Abse
IdentityIdentityAbs:y:0*
T0*1
_output_shapes
:џџџџџџџџџЗ2

Identity"
identityIdentity:output:0*0
_input_shapes
:џџџџџџџџџЗ:T P
1
_output_shapes
:џџџџџџџџџЗ

_user_specified_namex

Ј
5__inference_batch_normalization_3_layer_call_fn_22899

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityЂStatefulPartitionedCallД
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_204222
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 
_user_specified_nameinputs
Ф
s
K__inference_apply_filterbank_layer_call_and_return_conditional_losses_23240
x
tensordot_b
identityj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesu
Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          2
Tensordot/freeS
Tensordot/ShapeShapex*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axisб
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axisз
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axisА
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack
Tensordot/transpose	TransposexTensordot/concat:output:0*
T0*1
_output_shapes
:џџџџџџџџџЗ2
Tensordot/transpose
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2
Tensordot/Reshape
Tensordot/MatMulMatMulTensordot/Reshape:output:0tensordot_b*
T0*(
_output_shapes
:џџџџџџџџџ2
Tensordot/MatMulq
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axisН
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*1
_output_shapes
:џџџџџџџџџЗ2
	Tensordoty
transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             2
transpose/perm
	transpose	TransposeTensordot:output:0transpose/perm:output:0*
T0*1
_output_shapes
:џџџџџџџџџЗ2
	transposek
IdentityIdentitytranspose:y:0*
T0*1
_output_shapes
:џџџџџџџџџЗ2

Identity"
identityIdentity:output:0*<
_input_shapes+
):џџџџџџџџџЗ:
:T P
1
_output_shapes
:џџџџџџџџџЗ

_user_specified_namex:&"
 
_output_shapes
:


І
3__inference_batch_normalization_layer_call_fn_22442

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityЂStatefulPartitionedCallА
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_200432
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ј
ѓ
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_20815

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ : : : : :*
epsilon%o:*
is_training( 2
FusedBatchNormV3к
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:џџџџџџџџџ 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs

`
'__inference_dropout_layer_call_fn_23016

inputs
identityЂStatefulPartitionedCallи
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_209042
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*&
_input_shapes
:џџџџџџџџџ@22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
К
^
B__inference_flatten_layer_call_and_return_conditional_losses_20857

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ "  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:џџџџџџџџџD2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:џџџџџџџџџD2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ :W S
/
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs


P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_22771

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityЂAssignNewValueЂAssignNewValue_1ЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1и
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ" :::::*
epsilon%o:*
exponential_avg_factor%
з#<2
FusedBatchNormV3­
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValueЛ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1ў
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:џџџџџџџџџ" 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ" ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:џџџџџџџџџ" 
 
_user_specified_nameinputs

І
#__inference_signature_wrapper_21447
reshape_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27
identityЂStatefulPartitionedCallЧ
StatefulPartitionedCallStatefulPartitionedCallreshape_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27*)
Tin"
 2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*>
_read_only_resource_inputs 
	
*-
config_proto

CPU

GPU 2J 8 *)
f$R"
 __inference__wrapped_model_197642
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*І
_input_shapes
:џџџџџџџџџт	:
::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
)
_output_shapes
:џџџџџџџџџт	
'
_user_specified_namereshape_input:&"
 
_output_shapes
:


C
'__inference_reshape_layer_call_fn_22109

inputs
identityЦ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:џџџџџџџџџт	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_reshape_layer_call_and_return_conditional_losses_204502
PartitionedCallr
IdentityIdentityPartitionedCall:output:0*
T0*-
_output_shapes
:џџџџџџџџџт	2

Identity"
identityIdentity:output:0*(
_input_shapes
:џџџџџџџџџт	:Q M
)
_output_shapes
:џџџџџџџџџт	
 
_user_specified_nameinputs
Ћ
K
/__inference_max_pooling2d_1_layer_call_fn_20213

inputs
identityы
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_202072
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ЋU
ё	
E__inference_sequential_layer_call_and_return_conditional_losses_21313

inputs
melspectrogram_21237
batch_normalization_21240
batch_normalization_21242
batch_normalization_21244
batch_normalization_21246
conv2d_21249
conv2d_21251
batch_normalization_1_21255
batch_normalization_1_21257
batch_normalization_1_21259
batch_normalization_1_21261
conv2d_1_21264
conv2d_1_21266
batch_normalization_2_21270
batch_normalization_2_21272
batch_normalization_2_21274
batch_normalization_2_21276
conv2d_2_21279
conv2d_2_21281
batch_normalization_3_21285
batch_normalization_3_21287
batch_normalization_3_21289
batch_normalization_3_21291
dense_21295
dense_21297
dense_1_21301
dense_1_21303
dense_2_21307
dense_2_21309
identityЂ+batch_normalization/StatefulPartitionedCallЂ-batch_normalization_1/StatefulPartitionedCallЂ-batch_normalization_2/StatefulPartitionedCallЂ-batch_normalization_3/StatefulPartitionedCallЂconv2d/StatefulPartitionedCallЂ conv2d_1/StatefulPartitionedCallЂ conv2d_2/StatefulPartitionedCallЂdense/StatefulPartitionedCallЂdense_1/StatefulPartitionedCallЂdense_2/StatefulPartitionedCallж
reshape/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:џџџџџџџџџт	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_reshape_layer_call_and_return_conditional_losses_204502
reshape/PartitionedCall 
melspectrogram/PartitionedCallPartitionedCall reshape/PartitionedCall:output:0melspectrogram_21237*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:џџџџџџџџџЗ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_melspectrogram_layer_call_and_return_conditional_losses_199762 
melspectrogram/PartitionedCall­
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'melspectrogram/PartitionedCall:output:0batch_normalization_21240batch_normalization_21242batch_normalization_21244batch_normalization_21246*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:џџџџџџџџџЗ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_205122-
+batch_normalization/StatefulPartitionedCallП
conv2d/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0conv2d_21249conv2d_21251*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:џџџџџџџџџЗ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_205592 
conv2d/StatefulPartitionedCall
max_pooling2d/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџg@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_200912
max_pooling2d/PartitionedCallИ
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0batch_normalization_1_21255batch_normalization_1_21257batch_normalization_1_21259batch_normalization_1_21261*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџg@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_206132/
-batch_normalization_1/StatefulPartitionedCallЩ
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0conv2d_1_21264conv2d_1_21266*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџg@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_206602"
 conv2d_1/StatefulPartitionedCall
max_pooling2d_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ" * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_202072!
max_pooling2d_1/PartitionedCallК
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0batch_normalization_2_21270batch_normalization_2_21272batch_normalization_2_21274batch_normalization_2_21276*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ" *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_207142/
-batch_normalization_2/StatefulPartitionedCallЩ
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0conv2d_2_21279conv2d_2_21281*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ"  *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_conv2d_2_layer_call_and_return_conditional_losses_207612"
 conv2d_2/StatefulPartitionedCall
max_pooling2d_2/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_203232!
max_pooling2d_2/PartitionedCallК
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_2/PartitionedCall:output:0batch_normalization_3_21285batch_normalization_3_21287batch_normalization_3_21289batch_normalization_3_21291*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_208152/
-batch_normalization_3/StatefulPartitionedCall
flatten/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџD* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_208572
flatten/PartitionedCall
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_21295dense_21297*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_208762
dense/StatefulPartitionedCall№
dropout/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_209092
dropout/PartitionedCallІ
dense_1/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0dense_1_21301dense_1_21303*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_209332!
dense_1/StatefulPartitionedCallј
dropout_1/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_209662
dropout_1/PartitionedCallЈ
dense_2/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0dense_2_21307dense_2_21309*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_209902!
dense_2/StatefulPartitionedCall
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*І
_input_shapes
:џџџџџџџџџт	:
::::::::::::::::::::::::::::2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall:Q M
)
_output_shapes
:џџџџџџџџџт	
 
_user_specified_nameinputs:&"
 
_output_shapes
:

Ь

P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_22559

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityЂAssignNewValueЂAssignNewValue_1ЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:*
exponential_avg_factor%
з#<2
FusedBatchNormV3­
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValueЛ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs


P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_22919

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityЂAssignNewValueЂAssignNewValue_1ЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1и
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ : : : : :*
epsilon%o:*
exponential_avg_factor%
з#<2
FusedBatchNormV3­
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValueЛ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1ў
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:џџџџџџџџџ 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
Ь

P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_20391

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityЂAssignNewValueЂAssignNewValue_1ЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ : : : : :*
epsilon%o:*
exponential_avg_factor%
з#<2
FusedBatchNormV3­
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValueЛ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 
_user_specified_nameinputs
х
^
B__inference_reshape_layer_call_and_return_conditional_losses_22104

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicef
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB	 :т	2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/2 
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapeu
ReshapeReshapeinputsReshape/shape:output:0*
T0*-
_output_shapes
:џџџџџџџџџт	2	
Reshapej
IdentityIdentityReshape:output:0*
T0*-
_output_shapes
:џџџџџџџџџт	2

Identity"
identityIdentity:output:0*(
_input_shapes
:џџџџџџџџџт	:Q M
)
_output_shapes
:џџџџџџџџџт	
 
_user_specified_nameinputs
ВX
О

E__inference_sequential_layer_call_and_return_conditional_losses_21007
reshape_input
melspectrogram_20472
batch_normalization_20539
batch_normalization_20541
batch_normalization_20543
batch_normalization_20545
conv2d_20570
conv2d_20572
batch_normalization_1_20640
batch_normalization_1_20642
batch_normalization_1_20644
batch_normalization_1_20646
conv2d_1_20671
conv2d_1_20673
batch_normalization_2_20741
batch_normalization_2_20743
batch_normalization_2_20745
batch_normalization_2_20747
conv2d_2_20772
conv2d_2_20774
batch_normalization_3_20842
batch_normalization_3_20844
batch_normalization_3_20846
batch_normalization_3_20848
dense_20887
dense_20889
dense_1_20944
dense_1_20946
dense_2_21001
dense_2_21003
identityЂ+batch_normalization/StatefulPartitionedCallЂ-batch_normalization_1/StatefulPartitionedCallЂ-batch_normalization_2/StatefulPartitionedCallЂ-batch_normalization_3/StatefulPartitionedCallЂconv2d/StatefulPartitionedCallЂ conv2d_1/StatefulPartitionedCallЂ conv2d_2/StatefulPartitionedCallЂdense/StatefulPartitionedCallЂdense_1/StatefulPartitionedCallЂdense_2/StatefulPartitionedCallЂdropout/StatefulPartitionedCallЂ!dropout_1/StatefulPartitionedCallн
reshape/PartitionedCallPartitionedCallreshape_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:џџџџџџџџџт	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_reshape_layer_call_and_return_conditional_losses_204502
reshape/PartitionedCall 
melspectrogram/PartitionedCallPartitionedCall reshape/PartitionedCall:output:0melspectrogram_20472*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:џџџџџџџџџЗ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_melspectrogram_layer_call_and_return_conditional_losses_199602 
melspectrogram/PartitionedCallЋ
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'melspectrogram/PartitionedCall:output:0batch_normalization_20539batch_normalization_20541batch_normalization_20543batch_normalization_20545*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:џџџџџџџџџЗ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_204942-
+batch_normalization/StatefulPartitionedCallП
conv2d/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0conv2d_20570conv2d_20572*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:џџџџџџџџџЗ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_205592 
conv2d/StatefulPartitionedCall
max_pooling2d/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџg@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_200912
max_pooling2d/PartitionedCallЖ
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0batch_normalization_1_20640batch_normalization_1_20642batch_normalization_1_20644batch_normalization_1_20646*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџg@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_205952/
-batch_normalization_1/StatefulPartitionedCallЩ
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0conv2d_1_20671conv2d_1_20673*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџg@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_206602"
 conv2d_1/StatefulPartitionedCall
max_pooling2d_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ" * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_202072!
max_pooling2d_1/PartitionedCallИ
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0batch_normalization_2_20741batch_normalization_2_20743batch_normalization_2_20745batch_normalization_2_20747*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ" *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_206962/
-batch_normalization_2/StatefulPartitionedCallЩ
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0conv2d_2_20772conv2d_2_20774*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ"  *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_conv2d_2_layer_call_and_return_conditional_losses_207612"
 conv2d_2/StatefulPartitionedCall
max_pooling2d_2/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_203232!
max_pooling2d_2/PartitionedCallИ
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_2/PartitionedCall:output:0batch_normalization_3_20842batch_normalization_3_20844batch_normalization_3_20846batch_normalization_3_20848*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_207972/
-batch_normalization_3/StatefulPartitionedCall
flatten/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџD* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_208572
flatten/PartitionedCall
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_20887dense_20889*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_208762
dense/StatefulPartitionedCall
dropout/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_209042!
dropout/StatefulPartitionedCallЎ
dense_1/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0dense_1_20944dense_1_20946*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_209332!
dense_1/StatefulPartitionedCallВ
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_209612#
!dropout_1/StatefulPartitionedCallА
dense_2/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0dense_2_21001dense_2_21003*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_209902!
dense_2/StatefulPartitionedCallЫ
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*І
_input_shapes
:џџџџџџџџџт	:
::::::::::::::::::::::::::::2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall:X T
)
_output_shapes
:џџџџџџџџџт	
'
_user_specified_namereshape_input:&"
 
_output_shapes
:

с

I__inference_melspectrogram_layer_call_and_return_conditional_losses_19976

inputs
apply_filterbank_19972
identityб
stft/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:џџџџџџџџџЗ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_stft_layer_call_and_return_conditional_losses_198772
stft/PartitionedCallї
magnitude/PartitionedCallPartitionedCallstft/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:џџџџџџџџџЗ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_magnitude_layer_call_and_return_conditional_losses_198902
magnitude/PartitionedCallЊ
 apply_filterbank/PartitionedCallPartitionedCall"magnitude/PartitionedCall:output:0apply_filterbank_19972*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:џџџџџџџџџЗ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_apply_filterbank_layer_call_and_return_conditional_losses_199262"
 apply_filterbank/PartitionedCall
IdentityIdentity)apply_filterbank/PartitionedCall:output:0*
T0*1
_output_shapes
:џџџџџџџџџЗ2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:џџџџџџџџџт	:
:U Q
-
_output_shapes
:џџџџџџџџџт	
 
_user_specified_nameinputs:&"
 
_output_shapes
:

Ъ

N__inference_batch_normalization_layer_call_and_return_conditional_losses_22411

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityЂAssignNewValueЂAssignNewValue_1ЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:*
exponential_avg_factor%
з#<2
FusedBatchNormV3­
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValueЛ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ў
ё
N__inference_batch_normalization_layer_call_and_return_conditional_losses_20512

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ь
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:џџџџџџџџџЗ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3м
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*1
_output_shapes
:џџџџџџџџџЗ2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:џџџџџџџџџЗ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:џџџџџџџџџЗ
 
_user_specified_nameinputs
э

I__inference_melspectrogram_layer_call_and_return_conditional_losses_19939

stft_input
apply_filterbank_19935
identityе
stft/PartitionedCallPartitionedCall
stft_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:џџџџџџџџџЗ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_stft_layer_call_and_return_conditional_losses_198772
stft/PartitionedCallї
magnitude/PartitionedCallPartitionedCallstft/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:џџџџџџџџџЗ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_magnitude_layer_call_and_return_conditional_losses_198902
magnitude/PartitionedCallЊ
 apply_filterbank/PartitionedCallPartitionedCall"magnitude/PartitionedCall:output:0apply_filterbank_19935*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:џџџџџџџџџЗ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_apply_filterbank_layer_call_and_return_conditional_losses_199262"
 apply_filterbank/PartitionedCall
IdentityIdentity)apply_filterbank/PartitionedCall:output:0*
T0*1
_output_shapes
:џџџџџџџџџЗ2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:џџџџџџџџџт	:
:Y U
-
_output_shapes
:џџџџџџџџџт	
$
_user_specified_name
stft_input:&"
 
_output_shapes
:

Х
`
B__inference_dropout_layer_call_and_return_conditional_losses_20909

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:џџџџџџџџџ@2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:џџџџџџџџџ@:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs


N__inference_batch_normalization_layer_call_and_return_conditional_losses_22475

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityЂAssignNewValueЂAssignNewValue_1ЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1к
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:џџџџџџџџџЗ:::::*
epsilon%o:*
exponential_avg_factor%
з#<2
FusedBatchNormV3­
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValueЛ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*1
_output_shapes
:џџџџџџџџџЗ2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:џџџџџџџџџЗ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:џџџџџџџџџЗ
 
_user_specified_nameinputs
М
"
__inference__traced_save_23502
file_prefix8
4savev2_batch_normalization_gamma_read_readvariableop7
3savev2_batch_normalization_beta_read_readvariableop>
:savev2_batch_normalization_moving_mean_read_readvariableopB
>savev2_batch_normalization_moving_variance_read_readvariableop,
(savev2_conv2d_kernel_read_readvariableop*
&savev2_conv2d_bias_read_readvariableop:
6savev2_batch_normalization_1_gamma_read_readvariableop9
5savev2_batch_normalization_1_beta_read_readvariableop@
<savev2_batch_normalization_1_moving_mean_read_readvariableopD
@savev2_batch_normalization_1_moving_variance_read_readvariableop.
*savev2_conv2d_1_kernel_read_readvariableop,
(savev2_conv2d_1_bias_read_readvariableop:
6savev2_batch_normalization_2_gamma_read_readvariableop9
5savev2_batch_normalization_2_beta_read_readvariableop@
<savev2_batch_normalization_2_moving_mean_read_readvariableopD
@savev2_batch_normalization_2_moving_variance_read_readvariableop.
*savev2_conv2d_2_kernel_read_readvariableop,
(savev2_conv2d_2_bias_read_readvariableop:
6savev2_batch_normalization_3_gamma_read_readvariableop9
5savev2_batch_normalization_3_beta_read_readvariableop@
<savev2_batch_normalization_3_moving_mean_read_readvariableopD
@savev2_batch_normalization_3_moving_variance_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop-
)savev2_dense_2_kernel_read_readvariableop+
'savev2_dense_2_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop?
;savev2_adam_batch_normalization_gamma_m_read_readvariableop>
:savev2_adam_batch_normalization_beta_m_read_readvariableop3
/savev2_adam_conv2d_kernel_m_read_readvariableop1
-savev2_adam_conv2d_bias_m_read_readvariableopA
=savev2_adam_batch_normalization_1_gamma_m_read_readvariableop@
<savev2_adam_batch_normalization_1_beta_m_read_readvariableop5
1savev2_adam_conv2d_1_kernel_m_read_readvariableop3
/savev2_adam_conv2d_1_bias_m_read_readvariableopA
=savev2_adam_batch_normalization_2_gamma_m_read_readvariableop@
<savev2_adam_batch_normalization_2_beta_m_read_readvariableop5
1savev2_adam_conv2d_2_kernel_m_read_readvariableop3
/savev2_adam_conv2d_2_bias_m_read_readvariableopA
=savev2_adam_batch_normalization_3_gamma_m_read_readvariableop@
<savev2_adam_batch_normalization_3_beta_m_read_readvariableop2
.savev2_adam_dense_kernel_m_read_readvariableop0
,savev2_adam_dense_bias_m_read_readvariableop4
0savev2_adam_dense_1_kernel_m_read_readvariableop2
.savev2_adam_dense_1_bias_m_read_readvariableop4
0savev2_adam_dense_2_kernel_m_read_readvariableop2
.savev2_adam_dense_2_bias_m_read_readvariableop?
;savev2_adam_batch_normalization_gamma_v_read_readvariableop>
:savev2_adam_batch_normalization_beta_v_read_readvariableop3
/savev2_adam_conv2d_kernel_v_read_readvariableop1
-savev2_adam_conv2d_bias_v_read_readvariableopA
=savev2_adam_batch_normalization_1_gamma_v_read_readvariableop@
<savev2_adam_batch_normalization_1_beta_v_read_readvariableop5
1savev2_adam_conv2d_1_kernel_v_read_readvariableop3
/savev2_adam_conv2d_1_bias_v_read_readvariableopA
=savev2_adam_batch_normalization_2_gamma_v_read_readvariableop@
<savev2_adam_batch_normalization_2_beta_v_read_readvariableop5
1savev2_adam_conv2d_2_kernel_v_read_readvariableop3
/savev2_adam_conv2d_2_bias_v_read_readvariableopA
=savev2_adam_batch_normalization_3_gamma_v_read_readvariableop@
<savev2_adam_batch_normalization_3_beta_v_read_readvariableop2
.savev2_adam_dense_kernel_v_read_readvariableop0
,savev2_adam_dense_bias_v_read_readvariableop4
0savev2_adam_dense_1_kernel_v_read_readvariableop2
.savev2_adam_dense_1_bias_v_read_readvariableop4
0savev2_adam_dense_2_kernel_v_read_readvariableop2
.savev2_adam_dense_2_bias_v_read_readvariableop
savev2_const_1

identity_1ЂMergeV2Checkpoints
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shardІ
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename+
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:N*
dtype0* *
value*B*NB5layer_with_weights-0/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-0/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-0/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesЇ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:N*
dtype0*Б
valueЇBЄNB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesџ 
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:04savev2_batch_normalization_gamma_read_readvariableop3savev2_batch_normalization_beta_read_readvariableop:savev2_batch_normalization_moving_mean_read_readvariableop>savev2_batch_normalization_moving_variance_read_readvariableop(savev2_conv2d_kernel_read_readvariableop&savev2_conv2d_bias_read_readvariableop6savev2_batch_normalization_1_gamma_read_readvariableop5savev2_batch_normalization_1_beta_read_readvariableop<savev2_batch_normalization_1_moving_mean_read_readvariableop@savev2_batch_normalization_1_moving_variance_read_readvariableop*savev2_conv2d_1_kernel_read_readvariableop(savev2_conv2d_1_bias_read_readvariableop6savev2_batch_normalization_2_gamma_read_readvariableop5savev2_batch_normalization_2_beta_read_readvariableop<savev2_batch_normalization_2_moving_mean_read_readvariableop@savev2_batch_normalization_2_moving_variance_read_readvariableop*savev2_conv2d_2_kernel_read_readvariableop(savev2_conv2d_2_bias_read_readvariableop6savev2_batch_normalization_3_gamma_read_readvariableop5savev2_batch_normalization_3_beta_read_readvariableop<savev2_batch_normalization_3_moving_mean_read_readvariableop@savev2_batch_normalization_3_moving_variance_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop;savev2_adam_batch_normalization_gamma_m_read_readvariableop:savev2_adam_batch_normalization_beta_m_read_readvariableop/savev2_adam_conv2d_kernel_m_read_readvariableop-savev2_adam_conv2d_bias_m_read_readvariableop=savev2_adam_batch_normalization_1_gamma_m_read_readvariableop<savev2_adam_batch_normalization_1_beta_m_read_readvariableop1savev2_adam_conv2d_1_kernel_m_read_readvariableop/savev2_adam_conv2d_1_bias_m_read_readvariableop=savev2_adam_batch_normalization_2_gamma_m_read_readvariableop<savev2_adam_batch_normalization_2_beta_m_read_readvariableop1savev2_adam_conv2d_2_kernel_m_read_readvariableop/savev2_adam_conv2d_2_bias_m_read_readvariableop=savev2_adam_batch_normalization_3_gamma_m_read_readvariableop<savev2_adam_batch_normalization_3_beta_m_read_readvariableop.savev2_adam_dense_kernel_m_read_readvariableop,savev2_adam_dense_bias_m_read_readvariableop0savev2_adam_dense_1_kernel_m_read_readvariableop.savev2_adam_dense_1_bias_m_read_readvariableop0savev2_adam_dense_2_kernel_m_read_readvariableop.savev2_adam_dense_2_bias_m_read_readvariableop;savev2_adam_batch_normalization_gamma_v_read_readvariableop:savev2_adam_batch_normalization_beta_v_read_readvariableop/savev2_adam_conv2d_kernel_v_read_readvariableop-savev2_adam_conv2d_bias_v_read_readvariableop=savev2_adam_batch_normalization_1_gamma_v_read_readvariableop<savev2_adam_batch_normalization_1_beta_v_read_readvariableop1savev2_adam_conv2d_1_kernel_v_read_readvariableop/savev2_adam_conv2d_1_bias_v_read_readvariableop=savev2_adam_batch_normalization_2_gamma_v_read_readvariableop<savev2_adam_batch_normalization_2_beta_v_read_readvariableop1savev2_adam_conv2d_2_kernel_v_read_readvariableop/savev2_adam_conv2d_2_bias_v_read_readvariableop=savev2_adam_batch_normalization_3_gamma_v_read_readvariableop<savev2_adam_batch_normalization_3_beta_v_read_readvariableop.savev2_adam_dense_kernel_v_read_readvariableop,savev2_adam_dense_bias_v_read_readvariableop0savev2_adam_dense_1_kernel_v_read_readvariableop.savev2_adam_dense_1_bias_v_read_readvariableop0savev2_adam_dense_2_kernel_v_read_readvariableop.savev2_adam_dense_2_bias_v_read_readvariableopsavev2_const_1"/device:CPU:0*
_output_shapes
 *\
dtypesR
P2N	2
SaveV2К
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesЁ
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*ж
_input_shapesФ
С: ::::::::::::::::: : : : : : :	D@:@:@:::: : : : : : : : : ::::::::::: : : : :	D@:@:@:::::::::::::: : : : :	D@:@:@:::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 	

_output_shapes
:: 


_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :%!

_output_shapes
:	D@: 

_output_shapes
:@:$ 

_output_shapes

:@: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: :$

_output_shapes
: :%

_output_shapes
: : &

_output_shapes
:: '

_output_shapes
::,((
&
_output_shapes
:: )

_output_shapes
:: *

_output_shapes
:: +

_output_shapes
::,,(
&
_output_shapes
:: -

_output_shapes
:: .

_output_shapes
:: /

_output_shapes
::,0(
&
_output_shapes
: : 1

_output_shapes
: : 2

_output_shapes
: : 3

_output_shapes
: :%4!

_output_shapes
:	D@: 5

_output_shapes
:@:$6 

_output_shapes

:@: 7

_output_shapes
::$8 

_output_shapes

:: 9

_output_shapes
:: :

_output_shapes
:: ;

_output_shapes
::,<(
&
_output_shapes
:: =

_output_shapes
:: >

_output_shapes
:: ?

_output_shapes
::,@(
&
_output_shapes
:: A

_output_shapes
:: B

_output_shapes
:: C

_output_shapes
::,D(
&
_output_shapes
: : E

_output_shapes
: : F

_output_shapes
: : G

_output_shapes
: :%H!

_output_shapes
:	D@: I

_output_shapes
:@:$J 

_output_shapes

:@: K

_output_shapes
::$L 

_output_shapes

:: M

_output_shapes
::N

_output_shapes
: 

І
3__inference_batch_normalization_layer_call_fn_22455

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityЂStatefulPartitionedCallВ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_200742
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs

Ј
5__inference_batch_normalization_1_layer_call_fn_22590

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityЂStatefulPartitionedCallВ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_201592
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
и
|
'__inference_dense_1_layer_call_fn_23041

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallђ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_209332
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ@::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
Э

м
C__inference_conv2d_1_layer_call_and_return_conditional_losses_20660

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpЃ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџg@*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџg@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџg@2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:џџџџџџџџџg@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџg@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџg@
 
_user_specified_nameinputs
Ш
Љ
E__inference_sequential_layer_call_and_return_conditional_losses_21717

inputs/
+melspectrogram_apply_filterbank_tensordot_b/
+batch_normalization_readvariableop_resource1
-batch_normalization_readvariableop_1_resource@
<batch_normalization_fusedbatchnormv3_readvariableop_resourceB
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource1
-batch_normalization_1_readvariableop_resource3
/batch_normalization_1_readvariableop_1_resourceB
>batch_normalization_1_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource1
-batch_normalization_2_readvariableop_resource3
/batch_normalization_2_readvariableop_1_resourceB
>batch_normalization_2_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource+
'conv2d_2_conv2d_readvariableop_resource,
(conv2d_2_biasadd_readvariableop_resource1
-batch_normalization_3_readvariableop_resource3
/batch_normalization_3_readvariableop_1_resourceB
>batch_normalization_3_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource
identityЂ"batch_normalization/AssignNewValueЂ$batch_normalization/AssignNewValue_1Ђ3batch_normalization/FusedBatchNormV3/ReadVariableOpЂ5batch_normalization/FusedBatchNormV3/ReadVariableOp_1Ђ"batch_normalization/ReadVariableOpЂ$batch_normalization/ReadVariableOp_1Ђ$batch_normalization_1/AssignNewValueЂ&batch_normalization_1/AssignNewValue_1Ђ5batch_normalization_1/FusedBatchNormV3/ReadVariableOpЂ7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1Ђ$batch_normalization_1/ReadVariableOpЂ&batch_normalization_1/ReadVariableOp_1Ђ$batch_normalization_2/AssignNewValueЂ&batch_normalization_2/AssignNewValue_1Ђ5batch_normalization_2/FusedBatchNormV3/ReadVariableOpЂ7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1Ђ$batch_normalization_2/ReadVariableOpЂ&batch_normalization_2/ReadVariableOp_1Ђ$batch_normalization_3/AssignNewValueЂ&batch_normalization_3/AssignNewValue_1Ђ5batch_normalization_3/FusedBatchNormV3/ReadVariableOpЂ7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1Ђ$batch_normalization_3/ReadVariableOpЂ&batch_normalization_3/ReadVariableOp_1Ђconv2d/BiasAdd/ReadVariableOpЂconv2d/Conv2D/ReadVariableOpЂconv2d_1/BiasAdd/ReadVariableOpЂconv2d_1/Conv2D/ReadVariableOpЂconv2d_2/BiasAdd/ReadVariableOpЂconv2d_2/Conv2D/ReadVariableOpЂdense/BiasAdd/ReadVariableOpЂdense/MatMul/ReadVariableOpЂdense_1/BiasAdd/ReadVariableOpЂdense_1/MatMul/ReadVariableOpЂdense_2/BiasAdd/ReadVariableOpЂdense_2/MatMul/ReadVariableOpT
reshape/ShapeShapeinputs*
T0*
_output_shapes
:2
reshape/Shape
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape/strided_slice/stack
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_1
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_2
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape/strided_slicev
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB	 :т	2
reshape/Reshape/shape/1t
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/2Ш
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape/Reshape/shape
reshape/ReshapeReshapeinputsreshape/Reshape/shape:output:0*
T0*-
_output_shapes
:џџџџџџџџџт	2
reshape/Reshape
"melspectrogram/stft/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2$
"melspectrogram/stft/transpose/permЪ
melspectrogram/stft/transpose	Transposereshape/Reshape:output:0+melspectrogram/stft/transpose/perm:output:0*
T0*-
_output_shapes
:џџџџџџџџџт	2
melspectrogram/stft/transposeЏ
4melspectrogram/stft/stft_tf.signal.stft/frame_lengthConst*
_output_shapes
: *
dtype0*
value
B :26
4melspectrogram/stft/stft_tf.signal.stft/frame_lengthЋ
2melspectrogram/stft/stft_tf.signal.stft/frame_stepConst*
_output_shapes
: *
dtype0*
value
B :24
2melspectrogram/stft/stft_tf.signal.stft/frame_stepЋ
2melspectrogram/stft/stft_tf.signal.stft/fft_lengthConst*
_output_shapes
: *
dtype0*
value
B :24
2melspectrogram/stft/stft_tf.signal.stft/fft_lengthГ
2melspectrogram/stft/stft_tf.signal.stft/frame/axisConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ24
2melspectrogram/stft/stft_tf.signal.stft/frame/axisЛ
3melspectrogram/stft/stft_tf.signal.stft/frame/ShapeShape!melspectrogram/stft/transpose:y:0*
T0*
_output_shapes
:25
3melspectrogram/stft/stft_tf.signal.stft/frame/ShapeЊ
2melspectrogram/stft/stft_tf.signal.stft/frame/RankConst*
_output_shapes
: *
dtype0*
value	B :24
2melspectrogram/stft/stft_tf.signal.stft/frame/RankИ
9melspectrogram/stft/stft_tf.signal.stft/frame/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2;
9melspectrogram/stft/stft_tf.signal.stft/frame/range/startИ
9melspectrogram/stft/stft_tf.signal.stft/frame/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2;
9melspectrogram/stft/stft_tf.signal.stft/frame/range/deltaд
3melspectrogram/stft/stft_tf.signal.stft/frame/rangeRangeBmelspectrogram/stft/stft_tf.signal.stft/frame/range/start:output:0;melspectrogram/stft/stft_tf.signal.stft/frame/Rank:output:0Bmelspectrogram/stft/stft_tf.signal.stft/frame/range/delta:output:0*
_output_shapes
:25
3melspectrogram/stft/stft_tf.signal.stft/frame/rangeй
Amelspectrogram/stft/stft_tf.signal.stft/frame/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2C
Amelspectrogram/stft/stft_tf.signal.stft/frame/strided_slice/stackд
Cmelspectrogram/stft/stft_tf.signal.stft/frame/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2E
Cmelspectrogram/stft/stft_tf.signal.stft/frame/strided_slice/stack_1д
Cmelspectrogram/stft/stft_tf.signal.stft/frame/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2E
Cmelspectrogram/stft/stft_tf.signal.stft/frame/strided_slice/stack_2і
;melspectrogram/stft/stft_tf.signal.stft/frame/strided_sliceStridedSlice<melspectrogram/stft/stft_tf.signal.stft/frame/range:output:0Jmelspectrogram/stft/stft_tf.signal.stft/frame/strided_slice/stack:output:0Lmelspectrogram/stft/stft_tf.signal.stft/frame/strided_slice/stack_1:output:0Lmelspectrogram/stft/stft_tf.signal.stft/frame/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2=
;melspectrogram/stft/stft_tf.signal.stft/frame/strided_sliceЌ
3melspectrogram/stft/stft_tf.signal.stft/frame/sub/yConst*
_output_shapes
: *
dtype0*
value	B :25
3melspectrogram/stft/stft_tf.signal.stft/frame/sub/y
1melspectrogram/stft/stft_tf.signal.stft/frame/subSub;melspectrogram/stft/stft_tf.signal.stft/frame/Rank:output:0<melspectrogram/stft/stft_tf.signal.stft/frame/sub/y:output:0*
T0*
_output_shapes
: 23
1melspectrogram/stft/stft_tf.signal.stft/frame/sub
3melspectrogram/stft/stft_tf.signal.stft/frame/sub_1Sub5melspectrogram/stft/stft_tf.signal.stft/frame/sub:z:0Dmelspectrogram/stft/stft_tf.signal.stft/frame/strided_slice:output:0*
T0*
_output_shapes
: 25
3melspectrogram/stft/stft_tf.signal.stft/frame/sub_1В
6melspectrogram/stft/stft_tf.signal.stft/frame/packed/1Const*
_output_shapes
: *
dtype0*
value	B :28
6melspectrogram/stft/stft_tf.signal.stft/frame/packed/1т
4melspectrogram/stft/stft_tf.signal.stft/frame/packedPackDmelspectrogram/stft/stft_tf.signal.stft/frame/strided_slice:output:0?melspectrogram/stft/stft_tf.signal.stft/frame/packed/1:output:07melspectrogram/stft/stft_tf.signal.stft/frame/sub_1:z:0*
N*
T0*
_output_shapes
:26
4melspectrogram/stft/stft_tf.signal.stft/frame/packedР
=melspectrogram/stft/stft_tf.signal.stft/frame/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2?
=melspectrogram/stft/stft_tf.signal.stft/frame/split/split_dim
3melspectrogram/stft/stft_tf.signal.stft/frame/splitSplitV<melspectrogram/stft/stft_tf.signal.stft/frame/Shape:output:0=melspectrogram/stft/stft_tf.signal.stft/frame/packed:output:0Fmelspectrogram/stft/stft_tf.signal.stft/frame/split/split_dim:output:0*
T0*

Tlen0*$
_output_shapes
::: *
	num_split25
3melspectrogram/stft/stft_tf.signal.stft/frame/splitН
;melspectrogram/stft/stft_tf.signal.stft/frame/Reshape/shapeConst*
_output_shapes
: *
dtype0*
valueB 2=
;melspectrogram/stft/stft_tf.signal.stft/frame/Reshape/shapeС
=melspectrogram/stft/stft_tf.signal.stft/frame/Reshape/shape_1Const*
_output_shapes
: *
dtype0*
valueB 2?
=melspectrogram/stft/stft_tf.signal.stft/frame/Reshape/shape_1 
5melspectrogram/stft/stft_tf.signal.stft/frame/ReshapeReshape<melspectrogram/stft/stft_tf.signal.stft/frame/split:output:1Fmelspectrogram/stft/stft_tf.signal.stft/frame/Reshape/shape_1:output:0*
T0*
_output_shapes
: 27
5melspectrogram/stft/stft_tf.signal.stft/frame/ReshapeЊ
2melspectrogram/stft/stft_tf.signal.stft/frame/SizeConst*
_output_shapes
: *
dtype0*
value	B :24
2melspectrogram/stft/stft_tf.signal.stft/frame/SizeЎ
4melspectrogram/stft/stft_tf.signal.stft/frame/Size_1Const*
_output_shapes
: *
dtype0*
value	B : 26
4melspectrogram/stft/stft_tf.signal.stft/frame/Size_1
3melspectrogram/stft/stft_tf.signal.stft/frame/sub_2Sub>melspectrogram/stft/stft_tf.signal.stft/frame/Reshape:output:0=melspectrogram/stft/stft_tf.signal.stft/frame_length:output:0*
T0*
_output_shapes
: 25
3melspectrogram/stft/stft_tf.signal.stft/frame/sub_2
6melspectrogram/stft/stft_tf.signal.stft/frame/floordivFloorDiv7melspectrogram/stft/stft_tf.signal.stft/frame/sub_2:z:0;melspectrogram/stft/stft_tf.signal.stft/frame_step:output:0*
T0*
_output_shapes
: 28
6melspectrogram/stft/stft_tf.signal.stft/frame/floordivЌ
3melspectrogram/stft/stft_tf.signal.stft/frame/add/xConst*
_output_shapes
: *
dtype0*
value	B :25
3melspectrogram/stft/stft_tf.signal.stft/frame/add/x
1melspectrogram/stft/stft_tf.signal.stft/frame/addAddV2<melspectrogram/stft/stft_tf.signal.stft/frame/add/x:output:0:melspectrogram/stft/stft_tf.signal.stft/frame/floordiv:z:0*
T0*
_output_shapes
: 23
1melspectrogram/stft/stft_tf.signal.stft/frame/addД
7melspectrogram/stft/stft_tf.signal.stft/frame/Maximum/xConst*
_output_shapes
: *
dtype0*
value	B : 29
7melspectrogram/stft/stft_tf.signal.stft/frame/Maximum/x
5melspectrogram/stft/stft_tf.signal.stft/frame/MaximumMaximum@melspectrogram/stft/stft_tf.signal.stft/frame/Maximum/x:output:05melspectrogram/stft/stft_tf.signal.stft/frame/add:z:0*
T0*
_output_shapes
: 27
5melspectrogram/stft/stft_tf.signal.stft/frame/MaximumЕ
7melspectrogram/stft/stft_tf.signal.stft/frame/gcd/ConstConst*
_output_shapes
: *
dtype0*
value
B :29
7melspectrogram/stft/stft_tf.signal.stft/frame/gcd/ConstЛ
:melspectrogram/stft/stft_tf.signal.stft/frame/floordiv_1/yConst*
_output_shapes
: *
dtype0*
value
B :2<
:melspectrogram/stft/stft_tf.signal.stft/frame/floordiv_1/yЅ
8melspectrogram/stft/stft_tf.signal.stft/frame/floordiv_1FloorDiv=melspectrogram/stft/stft_tf.signal.stft/frame_length:output:0Cmelspectrogram/stft/stft_tf.signal.stft/frame/floordiv_1/y:output:0*
T0*
_output_shapes
: 2:
8melspectrogram/stft/stft_tf.signal.stft/frame/floordiv_1Л
:melspectrogram/stft/stft_tf.signal.stft/frame/floordiv_2/yConst*
_output_shapes
: *
dtype0*
value
B :2<
:melspectrogram/stft/stft_tf.signal.stft/frame/floordiv_2/yЃ
8melspectrogram/stft/stft_tf.signal.stft/frame/floordiv_2FloorDiv;melspectrogram/stft/stft_tf.signal.stft/frame_step:output:0Cmelspectrogram/stft/stft_tf.signal.stft/frame/floordiv_2/y:output:0*
T0*
_output_shapes
: 2:
8melspectrogram/stft/stft_tf.signal.stft/frame/floordiv_2Л
:melspectrogram/stft/stft_tf.signal.stft/frame/floordiv_3/yConst*
_output_shapes
: *
dtype0*
value
B :2<
:melspectrogram/stft/stft_tf.signal.stft/frame/floordiv_3/yІ
8melspectrogram/stft/stft_tf.signal.stft/frame/floordiv_3FloorDiv>melspectrogram/stft/stft_tf.signal.stft/frame/Reshape:output:0Cmelspectrogram/stft/stft_tf.signal.stft/frame/floordiv_3/y:output:0*
T0*
_output_shapes
: 2:
8melspectrogram/stft/stft_tf.signal.stft/frame/floordiv_3­
3melspectrogram/stft/stft_tf.signal.stft/frame/mul/yConst*
_output_shapes
: *
dtype0*
value
B :25
3melspectrogram/stft/stft_tf.signal.stft/frame/mul/y
1melspectrogram/stft/stft_tf.signal.stft/frame/mulMul<melspectrogram/stft/stft_tf.signal.stft/frame/floordiv_3:z:0<melspectrogram/stft/stft_tf.signal.stft/frame/mul/y:output:0*
T0*
_output_shapes
: 23
1melspectrogram/stft/stft_tf.signal.stft/frame/mulы
=melspectrogram/stft/stft_tf.signal.stft/frame/concat/values_1Pack5melspectrogram/stft/stft_tf.signal.stft/frame/mul:z:0*
N*
T0*
_output_shapes
:2?
=melspectrogram/stft/stft_tf.signal.stft/frame/concat/values_1И
9melspectrogram/stft/stft_tf.signal.stft/frame/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2;
9melspectrogram/stft/stft_tf.signal.stft/frame/concat/axisЎ
4melspectrogram/stft/stft_tf.signal.stft/frame/concatConcatV2<melspectrogram/stft/stft_tf.signal.stft/frame/split:output:0Fmelspectrogram/stft/stft_tf.signal.stft/frame/concat/values_1:output:0<melspectrogram/stft/stft_tf.signal.stft/frame/split:output:2Bmelspectrogram/stft/stft_tf.signal.stft/frame/concat/axis:output:0*
N*
T0*
_output_shapes
:26
4melspectrogram/stft/stft_tf.signal.stft/frame/concatЩ
Amelspectrogram/stft/stft_tf.signal.stft/frame/concat_1/values_1/1Const*
_output_shapes
: *
dtype0*
value
B :2C
Amelspectrogram/stft/stft_tf.signal.stft/frame/concat_1/values_1/1Т
?melspectrogram/stft/stft_tf.signal.stft/frame/concat_1/values_1Pack<melspectrogram/stft/stft_tf.signal.stft/frame/floordiv_3:z:0Jmelspectrogram/stft/stft_tf.signal.stft/frame/concat_1/values_1/1:output:0*
N*
T0*
_output_shapes
:2A
?melspectrogram/stft/stft_tf.signal.stft/frame/concat_1/values_1М
;melspectrogram/stft/stft_tf.signal.stft/frame/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2=
;melspectrogram/stft/stft_tf.signal.stft/frame/concat_1/axisЖ
6melspectrogram/stft/stft_tf.signal.stft/frame/concat_1ConcatV2<melspectrogram/stft/stft_tf.signal.stft/frame/split:output:0Hmelspectrogram/stft/stft_tf.signal.stft/frame/concat_1/values_1:output:0<melspectrogram/stft/stft_tf.signal.stft/frame/split:output:2Dmelspectrogram/stft/stft_tf.signal.stft/frame/concat_1/axis:output:0*
N*
T0*
_output_shapes
:28
6melspectrogram/stft/stft_tf.signal.stft/frame/concat_1О
8melspectrogram/stft/stft_tf.signal.stft/frame/zeros_likeConst*
_output_shapes
:*
dtype0*
valueB: 2:
8melspectrogram/stft/stft_tf.signal.stft/frame/zeros_likeШ
=melspectrogram/stft/stft_tf.signal.stft/frame/ones_like/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2?
=melspectrogram/stft/stft_tf.signal.stft/frame/ones_like/ShapeР
=melspectrogram/stft/stft_tf.signal.stft/frame/ones_like/ConstConst*
_output_shapes
: *
dtype0*
value	B :2?
=melspectrogram/stft/stft_tf.signal.stft/frame/ones_like/ConstЏ
7melspectrogram/stft/stft_tf.signal.stft/frame/ones_likeFillFmelspectrogram/stft/stft_tf.signal.stft/frame/ones_like/Shape:output:0Fmelspectrogram/stft/stft_tf.signal.stft/frame/ones_like/Const:output:0*
T0*
_output_shapes
:29
7melspectrogram/stft/stft_tf.signal.stft/frame/ones_likeФ
:melspectrogram/stft/stft_tf.signal.stft/frame/StridedSliceStridedSlice!melspectrogram/stft/transpose:y:0Amelspectrogram/stft/stft_tf.signal.stft/frame/zeros_like:output:0=melspectrogram/stft/stft_tf.signal.stft/frame/concat:output:0@melspectrogram/stft/stft_tf.signal.stft/frame/ones_like:output:0*
Index0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ2<
:melspectrogram/stft/stft_tf.signal.stft/frame/StridedSliceа
7melspectrogram/stft/stft_tf.signal.stft/frame/Reshape_1ReshapeCmelspectrogram/stft/stft_tf.signal.stft/frame/StridedSlice:output:0?melspectrogram/stft/stft_tf.signal.stft/frame/concat_1:output:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ29
7melspectrogram/stft/stft_tf.signal.stft/frame/Reshape_1М
;melspectrogram/stft/stft_tf.signal.stft/frame/range_1/startConst*
_output_shapes
: *
dtype0*
value	B : 2=
;melspectrogram/stft/stft_tf.signal.stft/frame/range_1/startМ
;melspectrogram/stft/stft_tf.signal.stft/frame/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :2=
;melspectrogram/stft/stft_tf.signal.stft/frame/range_1/deltaу
5melspectrogram/stft/stft_tf.signal.stft/frame/range_1RangeDmelspectrogram/stft/stft_tf.signal.stft/frame/range_1/start:output:09melspectrogram/stft/stft_tf.signal.stft/frame/Maximum:z:0Dmelspectrogram/stft/stft_tf.signal.stft/frame/range_1/delta:output:0*#
_output_shapes
:џџџџџџџџџ27
5melspectrogram/stft/stft_tf.signal.stft/frame/range_1
3melspectrogram/stft/stft_tf.signal.stft/frame/mul_1Mul>melspectrogram/stft/stft_tf.signal.stft/frame/range_1:output:0<melspectrogram/stft/stft_tf.signal.stft/frame/floordiv_2:z:0*
T0*#
_output_shapes
:џџџџџџџџџ25
3melspectrogram/stft/stft_tf.signal.stft/frame/mul_1Ф
?melspectrogram/stft/stft_tf.signal.stft/frame/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2A
?melspectrogram/stft/stft_tf.signal.stft/frame/Reshape_2/shape/1Й
=melspectrogram/stft/stft_tf.signal.stft/frame/Reshape_2/shapePack9melspectrogram/stft/stft_tf.signal.stft/frame/Maximum:z:0Hmelspectrogram/stft/stft_tf.signal.stft/frame/Reshape_2/shape/1:output:0*
N*
T0*
_output_shapes
:2?
=melspectrogram/stft/stft_tf.signal.stft/frame/Reshape_2/shapeА
7melspectrogram/stft/stft_tf.signal.stft/frame/Reshape_2Reshape7melspectrogram/stft/stft_tf.signal.stft/frame/mul_1:z:0Fmelspectrogram/stft/stft_tf.signal.stft/frame/Reshape_2/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ29
7melspectrogram/stft/stft_tf.signal.stft/frame/Reshape_2М
;melspectrogram/stft/stft_tf.signal.stft/frame/range_2/startConst*
_output_shapes
: *
dtype0*
value	B : 2=
;melspectrogram/stft/stft_tf.signal.stft/frame/range_2/startМ
;melspectrogram/stft/stft_tf.signal.stft/frame/range_2/deltaConst*
_output_shapes
: *
dtype0*
value	B :2=
;melspectrogram/stft/stft_tf.signal.stft/frame/range_2/deltaн
5melspectrogram/stft/stft_tf.signal.stft/frame/range_2RangeDmelspectrogram/stft/stft_tf.signal.stft/frame/range_2/start:output:0<melspectrogram/stft/stft_tf.signal.stft/frame/floordiv_1:z:0Dmelspectrogram/stft/stft_tf.signal.stft/frame/range_2/delta:output:0*
_output_shapes
:27
5melspectrogram/stft/stft_tf.signal.stft/frame/range_2Ф
?melspectrogram/stft/stft_tf.signal.stft/frame/Reshape_3/shape/0Const*
_output_shapes
: *
dtype0*
value	B :2A
?melspectrogram/stft/stft_tf.signal.stft/frame/Reshape_3/shape/0М
=melspectrogram/stft/stft_tf.signal.stft/frame/Reshape_3/shapePackHmelspectrogram/stft/stft_tf.signal.stft/frame/Reshape_3/shape/0:output:0<melspectrogram/stft/stft_tf.signal.stft/frame/floordiv_1:z:0*
N*
T0*
_output_shapes
:2?
=melspectrogram/stft/stft_tf.signal.stft/frame/Reshape_3/shapeЎ
7melspectrogram/stft/stft_tf.signal.stft/frame/Reshape_3Reshape>melspectrogram/stft/stft_tf.signal.stft/frame/range_2:output:0Fmelspectrogram/stft/stft_tf.signal.stft/frame/Reshape_3/shape:output:0*
T0*
_output_shapes

:29
7melspectrogram/stft/stft_tf.signal.stft/frame/Reshape_3Љ
3melspectrogram/stft/stft_tf.signal.stft/frame/add_1AddV2@melspectrogram/stft/stft_tf.signal.stft/frame/Reshape_2:output:0@melspectrogram/stft/stft_tf.signal.stft/frame/Reshape_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ25
3melspectrogram/stft/stft_tf.signal.stft/frame/add_1Б
6melspectrogram/stft/stft_tf.signal.stft/frame/GatherV2GatherV2@melspectrogram/stft/stft_tf.signal.stft/frame/Reshape_1:output:07melspectrogram/stft/stft_tf.signal.stft/frame/add_1:z:0Dmelspectrogram/stft/stft_tf.signal.stft/frame/strided_slice:output:0*
Taxis0*
Tindices0*
Tparams0*F
_output_shapes4
2:0џџџџџџџџџџџџџџџџџџџџџџџџџџџ28
6melspectrogram/stft/stft_tf.signal.stft/frame/GatherV2В
?melspectrogram/stft/stft_tf.signal.stft/frame/concat_2/values_1Pack9melspectrogram/stft/stft_tf.signal.stft/frame/Maximum:z:0=melspectrogram/stft/stft_tf.signal.stft/frame_length:output:0*
N*
T0*
_output_shapes
:2A
?melspectrogram/stft/stft_tf.signal.stft/frame/concat_2/values_1М
;melspectrogram/stft/stft_tf.signal.stft/frame/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2=
;melspectrogram/stft/stft_tf.signal.stft/frame/concat_2/axisЖ
6melspectrogram/stft/stft_tf.signal.stft/frame/concat_2ConcatV2<melspectrogram/stft/stft_tf.signal.stft/frame/split:output:0Hmelspectrogram/stft/stft_tf.signal.stft/frame/concat_2/values_1:output:0<melspectrogram/stft/stft_tf.signal.stft/frame/split:output:2Dmelspectrogram/stft/stft_tf.signal.stft/frame/concat_2/axis:output:0*
N*
T0*
_output_shapes
:28
6melspectrogram/stft/stft_tf.signal.stft/frame/concat_2Л
7melspectrogram/stft/stft_tf.signal.stft/frame/Reshape_4Reshape?melspectrogram/stft/stft_tf.signal.stft/frame/GatherV2:output:0?melspectrogram/stft/stft_tf.signal.stft/frame/concat_2:output:0*
T0*1
_output_shapes
:џџџџџџџџџЗ29
7melspectrogram/stft/stft_tf.signal.stft/frame/Reshape_4О
<melspectrogram/stft/stft_tf.signal.stft/hann_window/periodicConst*
_output_shapes
: *
dtype0
*
value	B
 Z2>
<melspectrogram/stft/stft_tf.signal.stft/hann_window/periodicѓ
8melspectrogram/stft/stft_tf.signal.stft/hann_window/CastCastEmelspectrogram/stft/stft_tf.signal.stft/hann_window/periodic:output:0*

DstT0*

SrcT0
*
_output_shapes
: 2:
8melspectrogram/stft/stft_tf.signal.stft/hann_window/CastТ
>melspectrogram/stft/stft_tf.signal.stft/hann_window/FloorMod/yConst*
_output_shapes
: *
dtype0*
value	B :2@
>melspectrogram/stft/stft_tf.signal.stft/hann_window/FloorMod/yБ
<melspectrogram/stft/stft_tf.signal.stft/hann_window/FloorModFloorMod=melspectrogram/stft/stft_tf.signal.stft/frame_length:output:0Gmelspectrogram/stft/stft_tf.signal.stft/hann_window/FloorMod/y:output:0*
T0*
_output_shapes
: 2>
<melspectrogram/stft/stft_tf.signal.stft/hann_window/FloorModИ
9melspectrogram/stft/stft_tf.signal.stft/hann_window/sub/xConst*
_output_shapes
: *
dtype0*
value	B :2;
9melspectrogram/stft/stft_tf.signal.stft/hann_window/sub/x 
7melspectrogram/stft/stft_tf.signal.stft/hann_window/subSubBmelspectrogram/stft/stft_tf.signal.stft/hann_window/sub/x:output:0@melspectrogram/stft/stft_tf.signal.stft/hann_window/FloorMod:z:0*
T0*
_output_shapes
: 29
7melspectrogram/stft/stft_tf.signal.stft/hann_window/sub
7melspectrogram/stft/stft_tf.signal.stft/hann_window/mulMul<melspectrogram/stft/stft_tf.signal.stft/hann_window/Cast:y:0;melspectrogram/stft/stft_tf.signal.stft/hann_window/sub:z:0*
T0*
_output_shapes
: 29
7melspectrogram/stft/stft_tf.signal.stft/hann_window/mul
7melspectrogram/stft/stft_tf.signal.stft/hann_window/addAddV2=melspectrogram/stft/stft_tf.signal.stft/frame_length:output:0;melspectrogram/stft/stft_tf.signal.stft/hann_window/mul:z:0*
T0*
_output_shapes
: 29
7melspectrogram/stft/stft_tf.signal.stft/hann_window/addМ
;melspectrogram/stft/stft_tf.signal.stft/hann_window/sub_1/yConst*
_output_shapes
: *
dtype0*
value	B :2=
;melspectrogram/stft/stft_tf.signal.stft/hann_window/sub_1/yЁ
9melspectrogram/stft/stft_tf.signal.stft/hann_window/sub_1Sub;melspectrogram/stft/stft_tf.signal.stft/hann_window/add:z:0Dmelspectrogram/stft/stft_tf.signal.stft/hann_window/sub_1/y:output:0*
T0*
_output_shapes
: 2;
9melspectrogram/stft/stft_tf.signal.stft/hann_window/sub_1я
:melspectrogram/stft/stft_tf.signal.stft/hann_window/Cast_1Cast=melspectrogram/stft/stft_tf.signal.stft/hann_window/sub_1:z:0*

DstT0*

SrcT0*
_output_shapes
: 2<
:melspectrogram/stft/stft_tf.signal.stft/hann_window/Cast_1Ф
?melspectrogram/stft/stft_tf.signal.stft/hann_window/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2A
?melspectrogram/stft/stft_tf.signal.stft/hann_window/range/startФ
?melspectrogram/stft/stft_tf.signal.stft/hann_window/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2A
?melspectrogram/stft/stft_tf.signal.stft/hann_window/range/deltaя
9melspectrogram/stft/stft_tf.signal.stft/hann_window/rangeRangeHmelspectrogram/stft/stft_tf.signal.stft/hann_window/range/start:output:0=melspectrogram/stft/stft_tf.signal.stft/frame_length:output:0Hmelspectrogram/stft/stft_tf.signal.stft/hann_window/range/delta:output:0*
_output_shapes	
:2;
9melspectrogram/stft/stft_tf.signal.stft/hann_window/rangeљ
:melspectrogram/stft/stft_tf.signal.stft/hann_window/Cast_2CastBmelspectrogram/stft/stft_tf.signal.stft/hann_window/range:output:0*

DstT0*

SrcT0*
_output_shapes	
:2<
:melspectrogram/stft/stft_tf.signal.stft/hann_window/Cast_2Л
9melspectrogram/stft/stft_tf.signal.stft/hann_window/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *лЩ@2;
9melspectrogram/stft/stft_tf.signal.stft/hann_window/ConstЇ
9melspectrogram/stft/stft_tf.signal.stft/hann_window/mul_1MulBmelspectrogram/stft/stft_tf.signal.stft/hann_window/Const:output:0>melspectrogram/stft/stft_tf.signal.stft/hann_window/Cast_2:y:0*
T0*
_output_shapes	
:2;
9melspectrogram/stft/stft_tf.signal.stft/hann_window/mul_1Њ
;melspectrogram/stft/stft_tf.signal.stft/hann_window/truedivRealDiv=melspectrogram/stft/stft_tf.signal.stft/hann_window/mul_1:z:0>melspectrogram/stft/stft_tf.signal.stft/hann_window/Cast_1:y:0*
T0*
_output_shapes	
:2=
;melspectrogram/stft/stft_tf.signal.stft/hann_window/truedivр
7melspectrogram/stft/stft_tf.signal.stft/hann_window/CosCos?melspectrogram/stft/stft_tf.signal.stft/hann_window/truediv:z:0*
T0*
_output_shapes	
:29
7melspectrogram/stft/stft_tf.signal.stft/hann_window/CosП
;melspectrogram/stft/stft_tf.signal.stft/hann_window/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2=
;melspectrogram/stft/stft_tf.signal.stft/hann_window/mul_2/xІ
9melspectrogram/stft/stft_tf.signal.stft/hann_window/mul_2MulDmelspectrogram/stft/stft_tf.signal.stft/hann_window/mul_2/x:output:0;melspectrogram/stft/stft_tf.signal.stft/hann_window/Cos:y:0*
T0*
_output_shapes	
:2;
9melspectrogram/stft/stft_tf.signal.stft/hann_window/mul_2П
;melspectrogram/stft/stft_tf.signal.stft/hann_window/sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2=
;melspectrogram/stft/stft_tf.signal.stft/hann_window/sub_2/xЈ
9melspectrogram/stft/stft_tf.signal.stft/hann_window/sub_2SubDmelspectrogram/stft/stft_tf.signal.stft/hann_window/sub_2/x:output:0=melspectrogram/stft/stft_tf.signal.stft/hann_window/mul_2:z:0*
T0*
_output_shapes	
:2;
9melspectrogram/stft/stft_tf.signal.stft/hann_window/sub_2
+melspectrogram/stft/stft_tf.signal.stft/mulMul@melspectrogram/stft/stft_tf.signal.stft/frame/Reshape_4:output:0=melspectrogram/stft/stft_tf.signal.stft/hann_window/sub_2:z:0*
T0*1
_output_shapes
:џџџџџџџџџЗ2-
+melspectrogram/stft/stft_tf.signal.stft/mulн
3melspectrogram/stft/stft_tf.signal.stft/rfft/packedPack;melspectrogram/stft/stft_tf.signal.stft/fft_length:output:0*
N*
T0*
_output_shapes
:25
3melspectrogram/stft/stft_tf.signal.stft/rfft/packedН
7melspectrogram/stft/stft_tf.signal.stft/rfft/fft_lengthConst*
_output_shapes
:*
dtype0*
valueB:29
7melspectrogram/stft/stft_tf.signal.stft/rfft/fft_length
,melspectrogram/stft/stft_tf.signal.stft/rfftRFFT/melspectrogram/stft/stft_tf.signal.stft/mul:z:0@melspectrogram/stft/stft_tf.signal.stft/rfft/fft_length:output:0*1
_output_shapes
:џџџџџџџџџЗ2.
,melspectrogram/stft/stft_tf.signal.stft/rfftЅ
$melspectrogram/stft/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             2&
$melspectrogram/stft/transpose_1/permё
melspectrogram/stft/transpose_1	Transpose5melspectrogram/stft/stft_tf.signal.stft/rfft:output:0-melspectrogram/stft/transpose_1/perm:output:0*
T0*1
_output_shapes
:џџџџџџџџџЗ2!
melspectrogram/stft/transpose_1Ђ
melspectrogram/magnitude/Abs
ComplexAbs#melspectrogram/stft/transpose_1:y:0*1
_output_shapes
:џџџџџџџџџЗ2
melspectrogram/magnitude/AbsЊ
.melspectrogram/apply_filterbank/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:20
.melspectrogram/apply_filterbank/Tensordot/axesЕ
.melspectrogram/apply_filterbank/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          20
.melspectrogram/apply_filterbank/Tensordot/freeВ
/melspectrogram/apply_filterbank/Tensordot/ShapeShape melspectrogram/magnitude/Abs:y:0*
T0*
_output_shapes
:21
/melspectrogram/apply_filterbank/Tensordot/ShapeД
7melspectrogram/apply_filterbank/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 29
7melspectrogram/apply_filterbank/Tensordot/GatherV2/axisё
2melspectrogram/apply_filterbank/Tensordot/GatherV2GatherV28melspectrogram/apply_filterbank/Tensordot/Shape:output:07melspectrogram/apply_filterbank/Tensordot/free:output:0@melspectrogram/apply_filterbank/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:24
2melspectrogram/apply_filterbank/Tensordot/GatherV2И
9melspectrogram/apply_filterbank/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2;
9melspectrogram/apply_filterbank/Tensordot/GatherV2_1/axisї
4melspectrogram/apply_filterbank/Tensordot/GatherV2_1GatherV28melspectrogram/apply_filterbank/Tensordot/Shape:output:07melspectrogram/apply_filterbank/Tensordot/axes:output:0Bmelspectrogram/apply_filterbank/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:26
4melspectrogram/apply_filterbank/Tensordot/GatherV2_1Ќ
/melspectrogram/apply_filterbank/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 21
/melspectrogram/apply_filterbank/Tensordot/Const
.melspectrogram/apply_filterbank/Tensordot/ProdProd;melspectrogram/apply_filterbank/Tensordot/GatherV2:output:08melspectrogram/apply_filterbank/Tensordot/Const:output:0*
T0*
_output_shapes
: 20
.melspectrogram/apply_filterbank/Tensordot/ProdА
1melspectrogram/apply_filterbank/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 23
1melspectrogram/apply_filterbank/Tensordot/Const_1
0melspectrogram/apply_filterbank/Tensordot/Prod_1Prod=melspectrogram/apply_filterbank/Tensordot/GatherV2_1:output:0:melspectrogram/apply_filterbank/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 22
0melspectrogram/apply_filterbank/Tensordot/Prod_1А
5melspectrogram/apply_filterbank/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 27
5melspectrogram/apply_filterbank/Tensordot/concat/axisа
0melspectrogram/apply_filterbank/Tensordot/concatConcatV27melspectrogram/apply_filterbank/Tensordot/free:output:07melspectrogram/apply_filterbank/Tensordot/axes:output:0>melspectrogram/apply_filterbank/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:22
0melspectrogram/apply_filterbank/Tensordot/concat
/melspectrogram/apply_filterbank/Tensordot/stackPack7melspectrogram/apply_filterbank/Tensordot/Prod:output:09melspectrogram/apply_filterbank/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:21
/melspectrogram/apply_filterbank/Tensordot/stack
3melspectrogram/apply_filterbank/Tensordot/transpose	Transpose melspectrogram/magnitude/Abs:y:09melspectrogram/apply_filterbank/Tensordot/concat:output:0*
T0*1
_output_shapes
:џџџџџџџџџЗ25
3melspectrogram/apply_filterbank/Tensordot/transpose
1melspectrogram/apply_filterbank/Tensordot/ReshapeReshape7melspectrogram/apply_filterbank/Tensordot/transpose:y:08melspectrogram/apply_filterbank/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ23
1melspectrogram/apply_filterbank/Tensordot/Reshape
0melspectrogram/apply_filterbank/Tensordot/MatMulMatMul:melspectrogram/apply_filterbank/Tensordot/Reshape:output:0+melspectrogram_apply_filterbank_tensordot_b*
T0*(
_output_shapes
:џџџџџџџџџ22
0melspectrogram/apply_filterbank/Tensordot/MatMulБ
1melspectrogram/apply_filterbank/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:23
1melspectrogram/apply_filterbank/Tensordot/Const_2Д
7melspectrogram/apply_filterbank/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 29
7melspectrogram/apply_filterbank/Tensordot/concat_1/axisн
2melspectrogram/apply_filterbank/Tensordot/concat_1ConcatV2;melspectrogram/apply_filterbank/Tensordot/GatherV2:output:0:melspectrogram/apply_filterbank/Tensordot/Const_2:output:0@melspectrogram/apply_filterbank/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:24
2melspectrogram/apply_filterbank/Tensordot/concat_1
)melspectrogram/apply_filterbank/TensordotReshape:melspectrogram/apply_filterbank/Tensordot/MatMul:product:0;melspectrogram/apply_filterbank/Tensordot/concat_1:output:0*
T0*1
_output_shapes
:џџџџџџџџџЗ2+
)melspectrogram/apply_filterbank/TensordotЙ
.melspectrogram/apply_filterbank/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             20
.melspectrogram/apply_filterbank/transpose/perm
)melspectrogram/apply_filterbank/transpose	Transpose2melspectrogram/apply_filterbank/Tensordot:output:07melspectrogram/apply_filterbank/transpose/perm:output:0*
T0*1
_output_shapes
:џџџџџџџџџЗ2+
)melspectrogram/apply_filterbank/transposeА
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
:*
dtype02$
"batch_normalization/ReadVariableOpЖ
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
:*
dtype02&
$batch_normalization/ReadVariableOp_1у
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization/FusedBatchNormV3/ReadVariableOpщ
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype027
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1љ
$batch_normalization/FusedBatchNormV3FusedBatchNormV3-melspectrogram/apply_filterbank/transpose:y:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:џџџџџџџџџЗ:::::*
epsilon%o:*
exponential_avg_factor%
з#<2&
$batch_normalization/FusedBatchNormV3Ѕ
"batch_normalization/AssignNewValueAssignVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource1batch_normalization/FusedBatchNormV3:batch_mean:04^batch_normalization/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*O
_classE
CAloc:@batch_normalization/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02$
"batch_normalization/AssignNewValueГ
$batch_normalization/AssignNewValue_1AssignVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource5batch_normalization/FusedBatchNormV3:batch_variance:06^batch_normalization/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*Q
_classG
ECloc:@batch_normalization/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02&
$batch_normalization/AssignNewValue_1Њ
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
conv2d/Conv2D/ReadVariableOpм
conv2d/Conv2DConv2D(batch_normalization/FusedBatchNormV3:y:0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџЗ*
paddingSAME*
strides
2
conv2d/Conv2DЁ
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv2d/BiasAdd/ReadVariableOpІ
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџЗ2
conv2d/BiasAddw
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*1
_output_shapes
:џџџџџџџџџЗ2
conv2d/ReluС
max_pooling2d/MaxPoolMaxPoolconv2d/Relu:activations:0*/
_output_shapes
:џџџџџџџџџg@*
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPoolЖ
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes
:*
dtype02&
$batch_normalization_1/ReadVariableOpМ
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes
:*
dtype02(
&batch_normalization_1/ReadVariableOp_1щ
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpя
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype029
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1є
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3max_pooling2d/MaxPool:output:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџg@:::::*
epsilon%o:*
exponential_avg_factor%
з#<2(
&batch_normalization_1/FusedBatchNormV3Б
$batch_normalization_1/AssignNewValueAssignVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource3batch_normalization_1/FusedBatchNormV3:batch_mean:06^batch_normalization_1/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*Q
_classG
ECloc:@batch_normalization_1/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02&
$batch_normalization_1/AssignNewValueП
&batch_normalization_1/AssignNewValue_1AssignVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_1/FusedBatchNormV3:batch_variance:08^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*S
_classI
GEloc:@batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02(
&batch_normalization_1/AssignNewValue_1А
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02 
conv2d_1/Conv2D/ReadVariableOpт
conv2d_1/Conv2DConv2D*batch_normalization_1/FusedBatchNormV3:y:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџg@*
paddingSAME*
strides
2
conv2d_1/Conv2DЇ
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_1/BiasAdd/ReadVariableOpЌ
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџg@2
conv2d_1/BiasAdd{
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџg@2
conv2d_1/ReluЧ
max_pooling2d_1/MaxPoolMaxPoolconv2d_1/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ" *
ksize
*
paddingVALID*
strides
2
max_pooling2d_1/MaxPoolЖ
$batch_normalization_2/ReadVariableOpReadVariableOp-batch_normalization_2_readvariableop_resource*
_output_shapes
:*
dtype02&
$batch_normalization_2/ReadVariableOpМ
&batch_normalization_2/ReadVariableOp_1ReadVariableOp/batch_normalization_2_readvariableop_1_resource*
_output_shapes
:*
dtype02(
&batch_normalization_2/ReadVariableOp_1щ
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpя
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype029
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1і
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3 max_pooling2d_1/MaxPool:output:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ" :::::*
epsilon%o:*
exponential_avg_factor%
з#<2(
&batch_normalization_2/FusedBatchNormV3Б
$batch_normalization_2/AssignNewValueAssignVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource3batch_normalization_2/FusedBatchNormV3:batch_mean:06^batch_normalization_2/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*Q
_classG
ECloc:@batch_normalization_2/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02&
$batch_normalization_2/AssignNewValueП
&batch_normalization_2/AssignNewValue_1AssignVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_2/FusedBatchNormV3:batch_variance:08^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*S
_classI
GEloc:@batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02(
&batch_normalization_2/AssignNewValue_1А
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02 
conv2d_2/Conv2D/ReadVariableOpт
conv2d_2/Conv2DConv2D*batch_normalization_2/FusedBatchNormV3:y:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ"  *
paddingSAME*
strides
2
conv2d_2/Conv2DЇ
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_2/BiasAdd/ReadVariableOpЌ
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ"  2
conv2d_2/BiasAdd{
conv2d_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ"  2
conv2d_2/ReluЧ
max_pooling2d_2/MaxPoolMaxPoolconv2d_2/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ *
ksize
*
paddingVALID*
strides
2
max_pooling2d_2/MaxPoolЖ
$batch_normalization_3/ReadVariableOpReadVariableOp-batch_normalization_3_readvariableop_resource*
_output_shapes
: *
dtype02&
$batch_normalization_3/ReadVariableOpМ
&batch_normalization_3/ReadVariableOp_1ReadVariableOp/batch_normalization_3_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&batch_normalization_3/ReadVariableOp_1щ
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype027
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpя
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype029
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1і
&batch_normalization_3/FusedBatchNormV3FusedBatchNormV3 max_pooling2d_2/MaxPool:output:0,batch_normalization_3/ReadVariableOp:value:0.batch_normalization_3/ReadVariableOp_1:value:0=batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ : : : : :*
epsilon%o:*
exponential_avg_factor%
з#<2(
&batch_normalization_3/FusedBatchNormV3Б
$batch_normalization_3/AssignNewValueAssignVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource3batch_normalization_3/FusedBatchNormV3:batch_mean:06^batch_normalization_3/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*Q
_classG
ECloc:@batch_normalization_3/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02&
$batch_normalization_3/AssignNewValueП
&batch_normalization_3/AssignNewValue_1AssignVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_3/FusedBatchNormV3:batch_variance:08^batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*S
_classI
GEloc:@batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02(
&batch_normalization_3/AssignNewValue_1o
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ "  2
flatten/ConstЄ
flatten/ReshapeReshape*batch_normalization_3/FusedBatchNormV3:y:0flatten/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџD2
flatten/Reshape 
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	D@*
dtype02
dense/MatMul/ReadVariableOp
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
dense/MatMul
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
dense/BiasAdd/ReadVariableOp
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
dense/BiasAddj

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2

dense/Relus
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/dropout/Const
dropout/dropout/MulMuldense/Relu:activations:0dropout/dropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
dropout/dropout/Mulv
dropout/dropout/ShapeShapedense/Relu:activations:0*
T0*
_output_shapes
:2
dropout/dropout/ShapeЬ
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@*
dtype02.
,dropout/dropout/random_uniform/RandomUniform
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>2 
dropout/dropout/GreaterEqual/yо
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
dropout/dropout/GreaterEqual
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ@2
dropout/dropout/Cast
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
dropout/dropout/Mul_1Ѕ
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02
dense_1/MatMul/ReadVariableOp
dense_1/MatMulMatMuldropout/dropout/Mul_1:z:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_1/MatMulЄ
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_1/BiasAdd/ReadVariableOpЁ
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_1/BiasAddp
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_1/Reluw
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout_1/dropout/ConstЅ
dropout_1/dropout/MulMuldense_1/Relu:activations:0 dropout_1/dropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dropout_1/dropout/Mul|
dropout_1/dropout/ShapeShapedense_1/Relu:activations:0*
T0*
_output_shapes
:2
dropout_1/dropout/Shapeв
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*
dtype020
.dropout_1/dropout/random_uniform/RandomUniform
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>2"
 dropout_1/dropout/GreaterEqual/yц
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2 
dropout_1/dropout/GreaterEqual
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ2
dropout_1/dropout/CastЂ
dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dropout_1/dropout/Mul_1Ѕ
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_2/MatMul/ReadVariableOp 
dense_2/MatMulMatMuldropout_1/dropout/Mul_1:z:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_2/MatMulЄ
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_2/BiasAdd/ReadVariableOpЁ
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_2/BiasAddy
dense_2/SigmoidSigmoiddense_2/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_2/SigmoidЇ
IdentityIdentitydense_2/Sigmoid:y:0#^batch_normalization/AssignNewValue%^batch_normalization/AssignNewValue_14^batch_normalization/FusedBatchNormV3/ReadVariableOp6^batch_normalization/FusedBatchNormV3/ReadVariableOp_1#^batch_normalization/ReadVariableOp%^batch_normalization/ReadVariableOp_1%^batch_normalization_1/AssignNewValue'^batch_normalization_1/AssignNewValue_16^batch_normalization_1/FusedBatchNormV3/ReadVariableOp8^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_1/ReadVariableOp'^batch_normalization_1/ReadVariableOp_1%^batch_normalization_2/AssignNewValue'^batch_normalization_2/AssignNewValue_16^batch_normalization_2/FusedBatchNormV3/ReadVariableOp8^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_2/ReadVariableOp'^batch_normalization_2/ReadVariableOp_1%^batch_normalization_3/AssignNewValue'^batch_normalization_3/AssignNewValue_16^batch_normalization_3/FusedBatchNormV3/ReadVariableOp8^batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_3/ReadVariableOp'^batch_normalization_3/ReadVariableOp_1^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*І
_input_shapes
:џџџџџџџџџт	:
::::::::::::::::::::::::::::2H
"batch_normalization/AssignNewValue"batch_normalization/AssignNewValue2L
$batch_normalization/AssignNewValue_1$batch_normalization/AssignNewValue_12j
3batch_normalization/FusedBatchNormV3/ReadVariableOp3batch_normalization/FusedBatchNormV3/ReadVariableOp2n
5batch_normalization/FusedBatchNormV3/ReadVariableOp_15batch_normalization/FusedBatchNormV3/ReadVariableOp_12H
"batch_normalization/ReadVariableOp"batch_normalization/ReadVariableOp2L
$batch_normalization/ReadVariableOp_1$batch_normalization/ReadVariableOp_12L
$batch_normalization_1/AssignNewValue$batch_normalization_1/AssignNewValue2P
&batch_normalization_1/AssignNewValue_1&batch_normalization_1/AssignNewValue_12n
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp5batch_normalization_1/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_17batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_1/ReadVariableOp$batch_normalization_1/ReadVariableOp2P
&batch_normalization_1/ReadVariableOp_1&batch_normalization_1/ReadVariableOp_12L
$batch_normalization_2/AssignNewValue$batch_normalization_2/AssignNewValue2P
&batch_normalization_2/AssignNewValue_1&batch_normalization_2/AssignNewValue_12n
5batch_normalization_2/FusedBatchNormV3/ReadVariableOp5batch_normalization_2/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_17batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_2/ReadVariableOp$batch_normalization_2/ReadVariableOp2P
&batch_normalization_2/ReadVariableOp_1&batch_normalization_2/ReadVariableOp_12L
$batch_normalization_3/AssignNewValue$batch_normalization_3/AssignNewValue2P
&batch_normalization_3/AssignNewValue_1&batch_normalization_3/AssignNewValue_12n
5batch_normalization_3/FusedBatchNormV3/ReadVariableOp5batch_normalization_3/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_17batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_3/ReadVariableOp$batch_normalization_3/ReadVariableOp2P
&batch_normalization_3/ReadVariableOp_1&batch_normalization_3/ReadVariableOp_12>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp:Q M
)
_output_shapes
:џџџџџџџџџт	
 
_user_specified_nameinputs:&"
 
_output_shapes
:

з

к
A__inference_conv2d_layer_call_and_return_conditional_losses_20559

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpЅ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџЗ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџЗ2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:џџџџџџџџџЗ2
ReluЁ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:џџџџџџџџџЗ2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:џџџџџџџџџЗ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:џџџџџџџџџЗ
 
_user_specified_nameinputs

T
0__inference_apply_filterbank_layer_call_fn_23247
x
unknown
identityи
PartitionedCallPartitionedCallxunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:џџџџџџџџџЗ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_apply_filterbank_layer_call_and_return_conditional_losses_199262
PartitionedCallv
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:џџџџџџџџџЗ2

Identity"
identityIdentity:output:0*<
_input_shapes+
):џџџџџџџџџЗ:
:T P
1
_output_shapes
:џџџџџџџџџЗ

_user_specified_namex:&"
 
_output_shapes
:

х
^
B__inference_reshape_layer_call_and_return_conditional_losses_20450

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicef
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB	 :т	2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/2 
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapeu
ReshapeReshapeinputsReshape/shape:output:0*
T0*-
_output_shapes
:џџџџџџџџџт	2	
Reshapej
IdentityIdentityReshape:output:0*
T0*-
_output_shapes
:џџџџџџџџџт	2

Identity"
identityIdentity:output:0*(
_input_shapes
:џџџџџџџџџт	:Q M
)
_output_shapes
:џџџџџџџџџт	
 
_user_specified_nameinputs
ў

a
B__inference_dropout_layer_call_and_return_conditional_losses_20904

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeД
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>2
dropout/GreaterEqual/yО
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ@2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*&
_input_shapes
:џџџџџџџџџ@:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs

Ј
5__inference_batch_normalization_1_layer_call_fn_22603

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityЂStatefulPartitionedCallД
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_201902
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ї
I
-__inference_max_pooling2d_layer_call_fn_20097

inputs
identityщ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_200912
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ј
ѓ
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_20714

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ" :::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3к
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:џџџџџџџџџ" 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ" ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:џџџџџџџџџ" 
 
_user_specified_nameinputs
е
Ј
5__inference_batch_normalization_2_layer_call_fn_22815

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityЂStatefulPartitionedCallЂ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ" *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_207142
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:џџџџџџџџџ" 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ" ::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ" 
 
_user_specified_nameinputs


N__inference_batch_normalization_layer_call_and_return_conditional_losses_20494

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityЂAssignNewValueЂAssignNewValue_1ЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1к
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:џџџџџџџџџЗ:::::*
epsilon%o:*
exponential_avg_factor%
з#<2
FusedBatchNormV3­
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValueЛ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*1
_output_shapes
:џџџџџџџџџЗ2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:џџџџџџџџџЗ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:џџџџџџџџџЗ
 
_user_specified_nameinputs

f
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_20323

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ь	
л
B__inference_dense_1_layer_call_and_return_conditional_losses_20933

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
ТЬ

I__inference_melspectrogram_layer_call_and_return_conditional_losses_22243

inputs 
apply_filterbank_tensordot_b
identity
stft/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
stft/transpose/perm
stft/transpose	Transposeinputsstft/transpose/perm:output:0*
T0*-
_output_shapes
:џџџџџџџџџт	2
stft/transpose
%stft/stft_tf.signal.stft/frame_lengthConst*
_output_shapes
: *
dtype0*
value
B :2'
%stft/stft_tf.signal.stft/frame_length
#stft/stft_tf.signal.stft/frame_stepConst*
_output_shapes
: *
dtype0*
value
B :2%
#stft/stft_tf.signal.stft/frame_step
#stft/stft_tf.signal.stft/fft_lengthConst*
_output_shapes
: *
dtype0*
value
B :2%
#stft/stft_tf.signal.stft/fft_length
#stft/stft_tf.signal.stft/frame/axisConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2%
#stft/stft_tf.signal.stft/frame/axis
$stft/stft_tf.signal.stft/frame/ShapeShapestft/transpose:y:0*
T0*
_output_shapes
:2&
$stft/stft_tf.signal.stft/frame/Shape
#stft/stft_tf.signal.stft/frame/RankConst*
_output_shapes
: *
dtype0*
value	B :2%
#stft/stft_tf.signal.stft/frame/Rank
*stft/stft_tf.signal.stft/frame/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2,
*stft/stft_tf.signal.stft/frame/range/start
*stft/stft_tf.signal.stft/frame/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2,
*stft/stft_tf.signal.stft/frame/range/delta
$stft/stft_tf.signal.stft/frame/rangeRange3stft/stft_tf.signal.stft/frame/range/start:output:0,stft/stft_tf.signal.stft/frame/Rank:output:03stft/stft_tf.signal.stft/frame/range/delta:output:0*
_output_shapes
:2&
$stft/stft_tf.signal.stft/frame/rangeЛ
2stft/stft_tf.signal.stft/frame/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ24
2stft/stft_tf.signal.stft/frame/strided_slice/stackЖ
4stft/stft_tf.signal.stft/frame/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 26
4stft/stft_tf.signal.stft/frame/strided_slice/stack_1Ж
4stft/stft_tf.signal.stft/frame/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:26
4stft/stft_tf.signal.stft/frame/strided_slice/stack_2
,stft/stft_tf.signal.stft/frame/strided_sliceStridedSlice-stft/stft_tf.signal.stft/frame/range:output:0;stft/stft_tf.signal.stft/frame/strided_slice/stack:output:0=stft/stft_tf.signal.stft/frame/strided_slice/stack_1:output:0=stft/stft_tf.signal.stft/frame/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2.
,stft/stft_tf.signal.stft/frame/strided_slice
$stft/stft_tf.signal.stft/frame/sub/yConst*
_output_shapes
: *
dtype0*
value	B :2&
$stft/stft_tf.signal.stft/frame/sub/yЭ
"stft/stft_tf.signal.stft/frame/subSub,stft/stft_tf.signal.stft/frame/Rank:output:0-stft/stft_tf.signal.stft/frame/sub/y:output:0*
T0*
_output_shapes
: 2$
"stft/stft_tf.signal.stft/frame/subг
$stft/stft_tf.signal.stft/frame/sub_1Sub&stft/stft_tf.signal.stft/frame/sub:z:05stft/stft_tf.signal.stft/frame/strided_slice:output:0*
T0*
_output_shapes
: 2&
$stft/stft_tf.signal.stft/frame/sub_1
'stft/stft_tf.signal.stft/frame/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2)
'stft/stft_tf.signal.stft/frame/packed/1
%stft/stft_tf.signal.stft/frame/packedPack5stft/stft_tf.signal.stft/frame/strided_slice:output:00stft/stft_tf.signal.stft/frame/packed/1:output:0(stft/stft_tf.signal.stft/frame/sub_1:z:0*
N*
T0*
_output_shapes
:2'
%stft/stft_tf.signal.stft/frame/packedЂ
.stft/stft_tf.signal.stft/frame/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 20
.stft/stft_tf.signal.stft/frame/split/split_dimК
$stft/stft_tf.signal.stft/frame/splitSplitV-stft/stft_tf.signal.stft/frame/Shape:output:0.stft/stft_tf.signal.stft/frame/packed:output:07stft/stft_tf.signal.stft/frame/split/split_dim:output:0*
T0*

Tlen0*$
_output_shapes
::: *
	num_split2&
$stft/stft_tf.signal.stft/frame/split
,stft/stft_tf.signal.stft/frame/Reshape/shapeConst*
_output_shapes
: *
dtype0*
valueB 2.
,stft/stft_tf.signal.stft/frame/Reshape/shapeЃ
.stft/stft_tf.signal.stft/frame/Reshape/shape_1Const*
_output_shapes
: *
dtype0*
valueB 20
.stft/stft_tf.signal.stft/frame/Reshape/shape_1ф
&stft/stft_tf.signal.stft/frame/ReshapeReshape-stft/stft_tf.signal.stft/frame/split:output:17stft/stft_tf.signal.stft/frame/Reshape/shape_1:output:0*
T0*
_output_shapes
: 2(
&stft/stft_tf.signal.stft/frame/Reshape
#stft/stft_tf.signal.stft/frame/SizeConst*
_output_shapes
: *
dtype0*
value	B :2%
#stft/stft_tf.signal.stft/frame/Size
%stft/stft_tf.signal.stft/frame/Size_1Const*
_output_shapes
: *
dtype0*
value	B : 2'
%stft/stft_tf.signal.stft/frame/Size_1е
$stft/stft_tf.signal.stft/frame/sub_2Sub/stft/stft_tf.signal.stft/frame/Reshape:output:0.stft/stft_tf.signal.stft/frame_length:output:0*
T0*
_output_shapes
: 2&
$stft/stft_tf.signal.stft/frame/sub_2з
'stft/stft_tf.signal.stft/frame/floordivFloorDiv(stft/stft_tf.signal.stft/frame/sub_2:z:0,stft/stft_tf.signal.stft/frame_step:output:0*
T0*
_output_shapes
: 2)
'stft/stft_tf.signal.stft/frame/floordiv
$stft/stft_tf.signal.stft/frame/add/xConst*
_output_shapes
: *
dtype0*
value	B :2&
$stft/stft_tf.signal.stft/frame/add/xЮ
"stft/stft_tf.signal.stft/frame/addAddV2-stft/stft_tf.signal.stft/frame/add/x:output:0+stft/stft_tf.signal.stft/frame/floordiv:z:0*
T0*
_output_shapes
: 2$
"stft/stft_tf.signal.stft/frame/add
(stft/stft_tf.signal.stft/frame/Maximum/xConst*
_output_shapes
: *
dtype0*
value	B : 2*
(stft/stft_tf.signal.stft/frame/Maximum/xз
&stft/stft_tf.signal.stft/frame/MaximumMaximum1stft/stft_tf.signal.stft/frame/Maximum/x:output:0&stft/stft_tf.signal.stft/frame/add:z:0*
T0*
_output_shapes
: 2(
&stft/stft_tf.signal.stft/frame/Maximum
(stft/stft_tf.signal.stft/frame/gcd/ConstConst*
_output_shapes
: *
dtype0*
value
B :2*
(stft/stft_tf.signal.stft/frame/gcd/Const
+stft/stft_tf.signal.stft/frame/floordiv_1/yConst*
_output_shapes
: *
dtype0*
value
B :2-
+stft/stft_tf.signal.stft/frame/floordiv_1/yщ
)stft/stft_tf.signal.stft/frame/floordiv_1FloorDiv.stft/stft_tf.signal.stft/frame_length:output:04stft/stft_tf.signal.stft/frame/floordiv_1/y:output:0*
T0*
_output_shapes
: 2+
)stft/stft_tf.signal.stft/frame/floordiv_1
+stft/stft_tf.signal.stft/frame/floordiv_2/yConst*
_output_shapes
: *
dtype0*
value
B :2-
+stft/stft_tf.signal.stft/frame/floordiv_2/yч
)stft/stft_tf.signal.stft/frame/floordiv_2FloorDiv,stft/stft_tf.signal.stft/frame_step:output:04stft/stft_tf.signal.stft/frame/floordiv_2/y:output:0*
T0*
_output_shapes
: 2+
)stft/stft_tf.signal.stft/frame/floordiv_2
+stft/stft_tf.signal.stft/frame/floordiv_3/yConst*
_output_shapes
: *
dtype0*
value
B :2-
+stft/stft_tf.signal.stft/frame/floordiv_3/yъ
)stft/stft_tf.signal.stft/frame/floordiv_3FloorDiv/stft/stft_tf.signal.stft/frame/Reshape:output:04stft/stft_tf.signal.stft/frame/floordiv_3/y:output:0*
T0*
_output_shapes
: 2+
)stft/stft_tf.signal.stft/frame/floordiv_3
$stft/stft_tf.signal.stft/frame/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2&
$stft/stft_tf.signal.stft/frame/mul/yЮ
"stft/stft_tf.signal.stft/frame/mulMul-stft/stft_tf.signal.stft/frame/floordiv_3:z:0-stft/stft_tf.signal.stft/frame/mul/y:output:0*
T0*
_output_shapes
: 2$
"stft/stft_tf.signal.stft/frame/mulО
.stft/stft_tf.signal.stft/frame/concat/values_1Pack&stft/stft_tf.signal.stft/frame/mul:z:0*
N*
T0*
_output_shapes
:20
.stft/stft_tf.signal.stft/frame/concat/values_1
*stft/stft_tf.signal.stft/frame/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2,
*stft/stft_tf.signal.stft/frame/concat/axisд
%stft/stft_tf.signal.stft/frame/concatConcatV2-stft/stft_tf.signal.stft/frame/split:output:07stft/stft_tf.signal.stft/frame/concat/values_1:output:0-stft/stft_tf.signal.stft/frame/split:output:23stft/stft_tf.signal.stft/frame/concat/axis:output:0*
N*
T0*
_output_shapes
:2'
%stft/stft_tf.signal.stft/frame/concatЋ
2stft/stft_tf.signal.stft/frame/concat_1/values_1/1Const*
_output_shapes
: *
dtype0*
value
B :24
2stft/stft_tf.signal.stft/frame/concat_1/values_1/1
0stft/stft_tf.signal.stft/frame/concat_1/values_1Pack-stft/stft_tf.signal.stft/frame/floordiv_3:z:0;stft/stft_tf.signal.stft/frame/concat_1/values_1/1:output:0*
N*
T0*
_output_shapes
:22
0stft/stft_tf.signal.stft/frame/concat_1/values_1
,stft/stft_tf.signal.stft/frame/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,stft/stft_tf.signal.stft/frame/concat_1/axisм
'stft/stft_tf.signal.stft/frame/concat_1ConcatV2-stft/stft_tf.signal.stft/frame/split:output:09stft/stft_tf.signal.stft/frame/concat_1/values_1:output:0-stft/stft_tf.signal.stft/frame/split:output:25stft/stft_tf.signal.stft/frame/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2)
'stft/stft_tf.signal.stft/frame/concat_1 
)stft/stft_tf.signal.stft/frame/zeros_likeConst*
_output_shapes
:*
dtype0*
valueB: 2+
)stft/stft_tf.signal.stft/frame/zeros_likeЊ
.stft/stft_tf.signal.stft/frame/ones_like/ShapeConst*
_output_shapes
:*
dtype0*
valueB:20
.stft/stft_tf.signal.stft/frame/ones_like/ShapeЂ
.stft/stft_tf.signal.stft/frame/ones_like/ConstConst*
_output_shapes
: *
dtype0*
value	B :20
.stft/stft_tf.signal.stft/frame/ones_like/Constѓ
(stft/stft_tf.signal.stft/frame/ones_likeFill7stft/stft_tf.signal.stft/frame/ones_like/Shape:output:07stft/stft_tf.signal.stft/frame/ones_like/Const:output:0*
T0*
_output_shapes
:2*
(stft/stft_tf.signal.stft/frame/ones_likeъ
+stft/stft_tf.signal.stft/frame/StridedSliceStridedSlicestft/transpose:y:02stft/stft_tf.signal.stft/frame/zeros_like:output:0.stft/stft_tf.signal.stft/frame/concat:output:01stft/stft_tf.signal.stft/frame/ones_like:output:0*
Index0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ2-
+stft/stft_tf.signal.stft/frame/StridedSlice
(stft/stft_tf.signal.stft/frame/Reshape_1Reshape4stft/stft_tf.signal.stft/frame/StridedSlice:output:00stft/stft_tf.signal.stft/frame/concat_1:output:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2*
(stft/stft_tf.signal.stft/frame/Reshape_1
,stft/stft_tf.signal.stft/frame/range_1/startConst*
_output_shapes
: *
dtype0*
value	B : 2.
,stft/stft_tf.signal.stft/frame/range_1/start
,stft/stft_tf.signal.stft/frame/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :2.
,stft/stft_tf.signal.stft/frame/range_1/delta
&stft/stft_tf.signal.stft/frame/range_1Range5stft/stft_tf.signal.stft/frame/range_1/start:output:0*stft/stft_tf.signal.stft/frame/Maximum:z:05stft/stft_tf.signal.stft/frame/range_1/delta:output:0*#
_output_shapes
:џџџџџџџџџ2(
&stft/stft_tf.signal.stft/frame/range_1с
$stft/stft_tf.signal.stft/frame/mul_1Mul/stft/stft_tf.signal.stft/frame/range_1:output:0-stft/stft_tf.signal.stft/frame/floordiv_2:z:0*
T0*#
_output_shapes
:џџџџџџџџџ2&
$stft/stft_tf.signal.stft/frame/mul_1І
0stft/stft_tf.signal.stft/frame/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :22
0stft/stft_tf.signal.stft/frame/Reshape_2/shape/1§
.stft/stft_tf.signal.stft/frame/Reshape_2/shapePack*stft/stft_tf.signal.stft/frame/Maximum:z:09stft/stft_tf.signal.stft/frame/Reshape_2/shape/1:output:0*
N*
T0*
_output_shapes
:20
.stft/stft_tf.signal.stft/frame/Reshape_2/shapeє
(stft/stft_tf.signal.stft/frame/Reshape_2Reshape(stft/stft_tf.signal.stft/frame/mul_1:z:07stft/stft_tf.signal.stft/frame/Reshape_2/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2*
(stft/stft_tf.signal.stft/frame/Reshape_2
,stft/stft_tf.signal.stft/frame/range_2/startConst*
_output_shapes
: *
dtype0*
value	B : 2.
,stft/stft_tf.signal.stft/frame/range_2/start
,stft/stft_tf.signal.stft/frame/range_2/deltaConst*
_output_shapes
: *
dtype0*
value	B :2.
,stft/stft_tf.signal.stft/frame/range_2/delta
&stft/stft_tf.signal.stft/frame/range_2Range5stft/stft_tf.signal.stft/frame/range_2/start:output:0-stft/stft_tf.signal.stft/frame/floordiv_1:z:05stft/stft_tf.signal.stft/frame/range_2/delta:output:0*
_output_shapes
:2(
&stft/stft_tf.signal.stft/frame/range_2І
0stft/stft_tf.signal.stft/frame/Reshape_3/shape/0Const*
_output_shapes
: *
dtype0*
value	B :22
0stft/stft_tf.signal.stft/frame/Reshape_3/shape/0
.stft/stft_tf.signal.stft/frame/Reshape_3/shapePack9stft/stft_tf.signal.stft/frame/Reshape_3/shape/0:output:0-stft/stft_tf.signal.stft/frame/floordiv_1:z:0*
N*
T0*
_output_shapes
:20
.stft/stft_tf.signal.stft/frame/Reshape_3/shapeђ
(stft/stft_tf.signal.stft/frame/Reshape_3Reshape/stft/stft_tf.signal.stft/frame/range_2:output:07stft/stft_tf.signal.stft/frame/Reshape_3/shape:output:0*
T0*
_output_shapes

:2*
(stft/stft_tf.signal.stft/frame/Reshape_3э
$stft/stft_tf.signal.stft/frame/add_1AddV21stft/stft_tf.signal.stft/frame/Reshape_2:output:01stft/stft_tf.signal.stft/frame/Reshape_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2&
$stft/stft_tf.signal.stft/frame/add_1ц
'stft/stft_tf.signal.stft/frame/GatherV2GatherV21stft/stft_tf.signal.stft/frame/Reshape_1:output:0(stft/stft_tf.signal.stft/frame/add_1:z:05stft/stft_tf.signal.stft/frame/strided_slice:output:0*
Taxis0*
Tindices0*
Tparams0*F
_output_shapes4
2:0џџџџџџџџџџџџџџџџџџџџџџџџџџџ2)
'stft/stft_tf.signal.stft/frame/GatherV2і
0stft/stft_tf.signal.stft/frame/concat_2/values_1Pack*stft/stft_tf.signal.stft/frame/Maximum:z:0.stft/stft_tf.signal.stft/frame_length:output:0*
N*
T0*
_output_shapes
:22
0stft/stft_tf.signal.stft/frame/concat_2/values_1
,stft/stft_tf.signal.stft/frame/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,stft/stft_tf.signal.stft/frame/concat_2/axisм
'stft/stft_tf.signal.stft/frame/concat_2ConcatV2-stft/stft_tf.signal.stft/frame/split:output:09stft/stft_tf.signal.stft/frame/concat_2/values_1:output:0-stft/stft_tf.signal.stft/frame/split:output:25stft/stft_tf.signal.stft/frame/concat_2/axis:output:0*
N*
T0*
_output_shapes
:2)
'stft/stft_tf.signal.stft/frame/concat_2џ
(stft/stft_tf.signal.stft/frame/Reshape_4Reshape0stft/stft_tf.signal.stft/frame/GatherV2:output:00stft/stft_tf.signal.stft/frame/concat_2:output:0*
T0*1
_output_shapes
:џџџџџџџџџЗ2*
(stft/stft_tf.signal.stft/frame/Reshape_4 
-stft/stft_tf.signal.stft/hann_window/periodicConst*
_output_shapes
: *
dtype0
*
value	B
 Z2/
-stft/stft_tf.signal.stft/hann_window/periodicЦ
)stft/stft_tf.signal.stft/hann_window/CastCast6stft/stft_tf.signal.stft/hann_window/periodic:output:0*

DstT0*

SrcT0
*
_output_shapes
: 2+
)stft/stft_tf.signal.stft/hann_window/CastЄ
/stft/stft_tf.signal.stft/hann_window/FloorMod/yConst*
_output_shapes
: *
dtype0*
value	B :21
/stft/stft_tf.signal.stft/hann_window/FloorMod/yѕ
-stft/stft_tf.signal.stft/hann_window/FloorModFloorMod.stft/stft_tf.signal.stft/frame_length:output:08stft/stft_tf.signal.stft/hann_window/FloorMod/y:output:0*
T0*
_output_shapes
: 2/
-stft/stft_tf.signal.stft/hann_window/FloorMod
*stft/stft_tf.signal.stft/hann_window/sub/xConst*
_output_shapes
: *
dtype0*
value	B :2,
*stft/stft_tf.signal.stft/hann_window/sub/xф
(stft/stft_tf.signal.stft/hann_window/subSub3stft/stft_tf.signal.stft/hann_window/sub/x:output:01stft/stft_tf.signal.stft/hann_window/FloorMod:z:0*
T0*
_output_shapes
: 2*
(stft/stft_tf.signal.stft/hann_window/subй
(stft/stft_tf.signal.stft/hann_window/mulMul-stft/stft_tf.signal.stft/hann_window/Cast:y:0,stft/stft_tf.signal.stft/hann_window/sub:z:0*
T0*
_output_shapes
: 2*
(stft/stft_tf.signal.stft/hann_window/mulм
(stft/stft_tf.signal.stft/hann_window/addAddV2.stft/stft_tf.signal.stft/frame_length:output:0,stft/stft_tf.signal.stft/hann_window/mul:z:0*
T0*
_output_shapes
: 2*
(stft/stft_tf.signal.stft/hann_window/add
,stft/stft_tf.signal.stft/hann_window/sub_1/yConst*
_output_shapes
: *
dtype0*
value	B :2.
,stft/stft_tf.signal.stft/hann_window/sub_1/yх
*stft/stft_tf.signal.stft/hann_window/sub_1Sub,stft/stft_tf.signal.stft/hann_window/add:z:05stft/stft_tf.signal.stft/hann_window/sub_1/y:output:0*
T0*
_output_shapes
: 2,
*stft/stft_tf.signal.stft/hann_window/sub_1Т
+stft/stft_tf.signal.stft/hann_window/Cast_1Cast.stft/stft_tf.signal.stft/hann_window/sub_1:z:0*

DstT0*

SrcT0*
_output_shapes
: 2-
+stft/stft_tf.signal.stft/hann_window/Cast_1І
0stft/stft_tf.signal.stft/hann_window/range/startConst*
_output_shapes
: *
dtype0*
value	B : 22
0stft/stft_tf.signal.stft/hann_window/range/startІ
0stft/stft_tf.signal.stft/hann_window/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :22
0stft/stft_tf.signal.stft/hann_window/range/deltaЄ
*stft/stft_tf.signal.stft/hann_window/rangeRange9stft/stft_tf.signal.stft/hann_window/range/start:output:0.stft/stft_tf.signal.stft/frame_length:output:09stft/stft_tf.signal.stft/hann_window/range/delta:output:0*
_output_shapes	
:2,
*stft/stft_tf.signal.stft/hann_window/rangeЬ
+stft/stft_tf.signal.stft/hann_window/Cast_2Cast3stft/stft_tf.signal.stft/hann_window/range:output:0*

DstT0*

SrcT0*
_output_shapes	
:2-
+stft/stft_tf.signal.stft/hann_window/Cast_2
*stft/stft_tf.signal.stft/hann_window/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *лЩ@2,
*stft/stft_tf.signal.stft/hann_window/Constы
*stft/stft_tf.signal.stft/hann_window/mul_1Mul3stft/stft_tf.signal.stft/hann_window/Const:output:0/stft/stft_tf.signal.stft/hann_window/Cast_2:y:0*
T0*
_output_shapes	
:2,
*stft/stft_tf.signal.stft/hann_window/mul_1ю
,stft/stft_tf.signal.stft/hann_window/truedivRealDiv.stft/stft_tf.signal.stft/hann_window/mul_1:z:0/stft/stft_tf.signal.stft/hann_window/Cast_1:y:0*
T0*
_output_shapes	
:2.
,stft/stft_tf.signal.stft/hann_window/truedivГ
(stft/stft_tf.signal.stft/hann_window/CosCos0stft/stft_tf.signal.stft/hann_window/truediv:z:0*
T0*
_output_shapes	
:2*
(stft/stft_tf.signal.stft/hann_window/CosЁ
,stft/stft_tf.signal.stft/hann_window/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2.
,stft/stft_tf.signal.stft/hann_window/mul_2/xъ
*stft/stft_tf.signal.stft/hann_window/mul_2Mul5stft/stft_tf.signal.stft/hann_window/mul_2/x:output:0,stft/stft_tf.signal.stft/hann_window/Cos:y:0*
T0*
_output_shapes	
:2,
*stft/stft_tf.signal.stft/hann_window/mul_2Ё
,stft/stft_tf.signal.stft/hann_window/sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2.
,stft/stft_tf.signal.stft/hann_window/sub_2/xь
*stft/stft_tf.signal.stft/hann_window/sub_2Sub5stft/stft_tf.signal.stft/hann_window/sub_2/x:output:0.stft/stft_tf.signal.stft/hann_window/mul_2:z:0*
T0*
_output_shapes	
:2,
*stft/stft_tf.signal.stft/hann_window/sub_2т
stft/stft_tf.signal.stft/mulMul1stft/stft_tf.signal.stft/frame/Reshape_4:output:0.stft/stft_tf.signal.stft/hann_window/sub_2:z:0*
T0*1
_output_shapes
:џџџџџџџџџЗ2
stft/stft_tf.signal.stft/mulА
$stft/stft_tf.signal.stft/rfft/packedPack,stft/stft_tf.signal.stft/fft_length:output:0*
N*
T0*
_output_shapes
:2&
$stft/stft_tf.signal.stft/rfft/packed
(stft/stft_tf.signal.stft/rfft/fft_lengthConst*
_output_shapes
:*
dtype0*
valueB:2*
(stft/stft_tf.signal.stft/rfft/fft_lengthЮ
stft/stft_tf.signal.stft/rfftRFFT stft/stft_tf.signal.stft/mul:z:01stft/stft_tf.signal.stft/rfft/fft_length:output:0*1
_output_shapes
:џџџџџџџџџЗ2
stft/stft_tf.signal.stft/rfft
stft/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             2
stft/transpose_1/permЕ
stft/transpose_1	Transpose&stft/stft_tf.signal.stft/rfft:output:0stft/transpose_1/perm:output:0*
T0*1
_output_shapes
:џџџџџџџџџЗ2
stft/transpose_1u
magnitude/Abs
ComplexAbsstft/transpose_1:y:0*1
_output_shapes
:џџџџџџџџџЗ2
magnitude/Abs
apply_filterbank/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2!
apply_filterbank/Tensordot/axes
apply_filterbank/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          2!
apply_filterbank/Tensordot/free
 apply_filterbank/Tensordot/ShapeShapemagnitude/Abs:y:0*
T0*
_output_shapes
:2"
 apply_filterbank/Tensordot/Shape
(apply_filterbank/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2*
(apply_filterbank/Tensordot/GatherV2/axisІ
#apply_filterbank/Tensordot/GatherV2GatherV2)apply_filterbank/Tensordot/Shape:output:0(apply_filterbank/Tensordot/free:output:01apply_filterbank/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2%
#apply_filterbank/Tensordot/GatherV2
*apply_filterbank/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2,
*apply_filterbank/Tensordot/GatherV2_1/axisЌ
%apply_filterbank/Tensordot/GatherV2_1GatherV2)apply_filterbank/Tensordot/Shape:output:0(apply_filterbank/Tensordot/axes:output:03apply_filterbank/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2'
%apply_filterbank/Tensordot/GatherV2_1
 apply_filterbank/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 apply_filterbank/Tensordot/ConstФ
apply_filterbank/Tensordot/ProdProd,apply_filterbank/Tensordot/GatherV2:output:0)apply_filterbank/Tensordot/Const:output:0*
T0*
_output_shapes
: 2!
apply_filterbank/Tensordot/Prod
"apply_filterbank/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2$
"apply_filterbank/Tensordot/Const_1Ь
!apply_filterbank/Tensordot/Prod_1Prod.apply_filterbank/Tensordot/GatherV2_1:output:0+apply_filterbank/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2#
!apply_filterbank/Tensordot/Prod_1
&apply_filterbank/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2(
&apply_filterbank/Tensordot/concat/axis
!apply_filterbank/Tensordot/concatConcatV2(apply_filterbank/Tensordot/free:output:0(apply_filterbank/Tensordot/axes:output:0/apply_filterbank/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2#
!apply_filterbank/Tensordot/concatа
 apply_filterbank/Tensordot/stackPack(apply_filterbank/Tensordot/Prod:output:0*apply_filterbank/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2"
 apply_filterbank/Tensordot/stackд
$apply_filterbank/Tensordot/transpose	Transposemagnitude/Abs:y:0*apply_filterbank/Tensordot/concat:output:0*
T0*1
_output_shapes
:џџџџџџџџџЗ2&
$apply_filterbank/Tensordot/transposeу
"apply_filterbank/Tensordot/ReshapeReshape(apply_filterbank/Tensordot/transpose:y:0)apply_filterbank/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2$
"apply_filterbank/Tensordot/ReshapeЮ
!apply_filterbank/Tensordot/MatMulMatMul+apply_filterbank/Tensordot/Reshape:output:0apply_filterbank_tensordot_b*
T0*(
_output_shapes
:џџџџџџџџџ2#
!apply_filterbank/Tensordot/MatMul
"apply_filterbank/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"apply_filterbank/Tensordot/Const_2
(apply_filterbank/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2*
(apply_filterbank/Tensordot/concat_1/axis
#apply_filterbank/Tensordot/concat_1ConcatV2,apply_filterbank/Tensordot/GatherV2:output:0+apply_filterbank/Tensordot/Const_2:output:01apply_filterbank/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2%
#apply_filterbank/Tensordot/concat_1к
apply_filterbank/TensordotReshape+apply_filterbank/Tensordot/MatMul:product:0,apply_filterbank/Tensordot/concat_1:output:0*
T0*1
_output_shapes
:џџџџџџџџџЗ2
apply_filterbank/Tensordot
apply_filterbank/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             2!
apply_filterbank/transpose/permа
apply_filterbank/transpose	Transpose#apply_filterbank/Tensordot:output:0(apply_filterbank/transpose/perm:output:0*
T0*1
_output_shapes
:џџџџџџџџџЗ2
apply_filterbank/transpose|
IdentityIdentityapply_filterbank/transpose:y:0*
T0*1
_output_shapes
:џџџџџџџџџЗ2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:џџџџџџџџџт	:
:U Q
-
_output_shapes
:џџџџџџџџџт	
 
_user_specified_nameinputs:&"
 
_output_shapes
:

Ь

P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_22707

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityЂAssignNewValueЂAssignNewValue_1ЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:*
exponential_avg_factor%
з#<2
FusedBatchNormV3­
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValueЛ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
О
ё
N__inference_batch_normalization_layer_call_and_return_conditional_losses_22429

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1м
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3ь
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs

[
.__inference_melspectrogram_layer_call_fn_19981

stft_input
unknown
identityп
PartitionedCallPartitionedCall
stft_inputunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:џџџџџџџџџЗ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_melspectrogram_layer_call_and_return_conditional_losses_199762
PartitionedCallv
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:џџџџџџџџџЗ2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:џџџџџџџџџт	:
:Y U
-
_output_shapes
:џџџџџџџџџт	
$
_user_specified_name
stft_input:&"
 
_output_shapes
:

з

к
A__inference_conv2d_layer_call_and_return_conditional_losses_22530

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpЅ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџЗ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџЗ2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:џџџџџџџџџЗ2
ReluЁ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:џџџџџџџџџЗ2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:џџџџџџџџџЗ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:џџџџџџџџџЗ
 
_user_specified_nameinputs
г
Ј
5__inference_batch_normalization_3_layer_call_fn_22950

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityЂStatefulPartitionedCall 
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_207972
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:џџџџџџџџџ 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ ::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs

Ј
5__inference_batch_normalization_3_layer_call_fn_22886

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityЂStatefulPartitionedCallВ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_203912
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 
_user_specified_nameinputs
Х
`
B__inference_dropout_layer_call_and_return_conditional_losses_23011

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:џџџџџџџџџ@2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:џџџџџџџџџ@:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
ў
d
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_20091

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ж
z
%__inference_dense_layer_call_fn_22994

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCall№
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_208762
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџD::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџD
 
_user_specified_nameinputs


P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_22623

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityЂAssignNewValueЂAssignNewValue_1ЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1и
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџg@:::::*
epsilon%o:*
exponential_avg_factor%
з#<2
FusedBatchNormV3­
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValueЛ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1ў
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:џџџџџџџџџg@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџg@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:џџџџџџџџџg@
 
_user_specified_nameinputs
Р
ѓ
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_22725

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1м
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3ь
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs


P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_20797

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityЂAssignNewValueЂAssignNewValue_1ЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1и
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ : : : : :*
epsilon%o:*
exponential_avg_factor%
з#<2
FusedBatchNormV3­
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValueЛ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1ў
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:џџџџџџџџџ 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
ј
ѓ
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_22641

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџg@:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3к
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:џџџџџџџџџg@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџg@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:џџџџџџџџџg@
 
_user_specified_nameinputs
ј
ѓ
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_22937

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ : : : : :*
epsilon%o:*
is_training( 2
FusedBatchNormV3к
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:џџџџџџџџџ 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
е
Ј
5__inference_batch_normalization_3_layer_call_fn_22963

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityЂStatefulPartitionedCallЂ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_208152
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:џџџџџџџџџ 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ ::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs


P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_20696

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityЂAssignNewValueЂAssignNewValue_1ЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1и
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ" :::::*
epsilon%o:*
exponential_avg_factor%
з#<2
FusedBatchNormV3­
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValueЛ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1ў
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:џџџџџџџџџ" 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ" ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:џџџџџџџџџ" 
 
_user_specified_nameinputs
І
­
*__inference_sequential_layer_call_fn_21231
reshape_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27
identityЂStatefulPartitionedCallф
StatefulPartitionedCallStatefulPartitionedCallreshape_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27*)
Tin"
 2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*6
_read_only_resource_inputs
	*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_211702
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*І
_input_shapes
:џџџџџџџџџт	:
::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
)
_output_shapes
:џџџџџџџџџт	
'
_user_specified_namereshape_input:&"
 
_output_shapes
:

К
^
B__inference_flatten_layer_call_and_return_conditional_losses_22969

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ "  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:џџџџџџџџџD2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:џџџџџџџџџD2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ :W S
/
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
Ў
­
*__inference_sequential_layer_call_fn_21374
reshape_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27
identityЂStatefulPartitionedCallь
StatefulPartitionedCallStatefulPartitionedCallreshape_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27*)
Tin"
 2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*>
_read_only_resource_inputs 
	
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_213132
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*І
_input_shapes
:џџџџџџџџџт	:
::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
)
_output_shapes
:џџџџџџџџџт	
'
_user_specified_namereshape_input:&"
 
_output_shapes
:

ў
ё
N__inference_batch_normalization_layer_call_and_return_conditional_losses_22493

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ь
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:џџџџџџџџџЗ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3м
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*1
_output_shapes
:џџџџџџџџџЗ2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:џџџџџџџџџЗ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:џџџџџџџџџЗ
 
_user_specified_nameinputs
X
З

E__inference_sequential_layer_call_and_return_conditional_losses_21170

inputs
melspectrogram_21094
batch_normalization_21097
batch_normalization_21099
batch_normalization_21101
batch_normalization_21103
conv2d_21106
conv2d_21108
batch_normalization_1_21112
batch_normalization_1_21114
batch_normalization_1_21116
batch_normalization_1_21118
conv2d_1_21121
conv2d_1_21123
batch_normalization_2_21127
batch_normalization_2_21129
batch_normalization_2_21131
batch_normalization_2_21133
conv2d_2_21136
conv2d_2_21138
batch_normalization_3_21142
batch_normalization_3_21144
batch_normalization_3_21146
batch_normalization_3_21148
dense_21152
dense_21154
dense_1_21158
dense_1_21160
dense_2_21164
dense_2_21166
identityЂ+batch_normalization/StatefulPartitionedCallЂ-batch_normalization_1/StatefulPartitionedCallЂ-batch_normalization_2/StatefulPartitionedCallЂ-batch_normalization_3/StatefulPartitionedCallЂconv2d/StatefulPartitionedCallЂ conv2d_1/StatefulPartitionedCallЂ conv2d_2/StatefulPartitionedCallЂdense/StatefulPartitionedCallЂdense_1/StatefulPartitionedCallЂdense_2/StatefulPartitionedCallЂdropout/StatefulPartitionedCallЂ!dropout_1/StatefulPartitionedCallж
reshape/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:џџџџџџџџџт	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_reshape_layer_call_and_return_conditional_losses_204502
reshape/PartitionedCall 
melspectrogram/PartitionedCallPartitionedCall reshape/PartitionedCall:output:0melspectrogram_21094*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:џџџџџџџџџЗ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_melspectrogram_layer_call_and_return_conditional_losses_199602 
melspectrogram/PartitionedCallЋ
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'melspectrogram/PartitionedCall:output:0batch_normalization_21097batch_normalization_21099batch_normalization_21101batch_normalization_21103*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:џџџџџџџџџЗ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_204942-
+batch_normalization/StatefulPartitionedCallП
conv2d/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0conv2d_21106conv2d_21108*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:џџџџџџџџџЗ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_205592 
conv2d/StatefulPartitionedCall
max_pooling2d/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџg@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_200912
max_pooling2d/PartitionedCallЖ
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0batch_normalization_1_21112batch_normalization_1_21114batch_normalization_1_21116batch_normalization_1_21118*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџg@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_205952/
-batch_normalization_1/StatefulPartitionedCallЩ
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0conv2d_1_21121conv2d_1_21123*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџg@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_206602"
 conv2d_1/StatefulPartitionedCall
max_pooling2d_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ" * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_202072!
max_pooling2d_1/PartitionedCallИ
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0batch_normalization_2_21127batch_normalization_2_21129batch_normalization_2_21131batch_normalization_2_21133*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ" *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_206962/
-batch_normalization_2/StatefulPartitionedCallЩ
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0conv2d_2_21136conv2d_2_21138*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ"  *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_conv2d_2_layer_call_and_return_conditional_losses_207612"
 conv2d_2/StatefulPartitionedCall
max_pooling2d_2/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_203232!
max_pooling2d_2/PartitionedCallИ
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_2/PartitionedCall:output:0batch_normalization_3_21142batch_normalization_3_21144batch_normalization_3_21146batch_normalization_3_21148*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_207972/
-batch_normalization_3/StatefulPartitionedCall
flatten/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџD* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_208572
flatten/PartitionedCall
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_21152dense_21154*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_208762
dense/StatefulPartitionedCall
dropout/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_209042!
dropout/StatefulPartitionedCallЎ
dense_1/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0dense_1_21158dense_1_21160*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_209332!
dense_1/StatefulPartitionedCallВ
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_209612#
!dropout_1/StatefulPartitionedCallА
dense_2/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0dense_2_21164dense_2_21166*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_209902!
dense_2/StatefulPartitionedCallЫ
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*І
_input_shapes
:џџџџџџџџџт	:
::::::::::::::::::::::::::::2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall:Q M
)
_output_shapes
:џџџџџџџџџт	
 
_user_specified_nameinputs:&"
 
_output_shapes
:

Э

м
C__inference_conv2d_2_layer_call_and_return_conditional_losses_22826

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOpЃ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ"  *
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ"  2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ"  2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:џџџџџџџџџ"  2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџ" ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ" 
 
_user_specified_nameinputs

Ј
5__inference_batch_normalization_2_layer_call_fn_22738

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityЂStatefulPartitionedCallВ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_202752
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Р
ѓ
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_20422

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1м
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ : : : : :*
epsilon%o:*
is_training( 2
FusedBatchNormV3ь
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 
_user_specified_nameinputs

І
*__inference_sequential_layer_call_fn_22091

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27
identityЂStatefulPartitionedCallх
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27*)
Tin"
 2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*>
_read_only_resource_inputs 
	
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_213132
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*І
_input_shapes
:џџџџџџџџџт	:
::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
)
_output_shapes
:џџџџџџџџџт	
 
_user_specified_nameinputs:&"
 
_output_shapes
:

э	
й
@__inference_dense_layer_call_and_return_conditional_losses_22985

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	D@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџD::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:џџџџџџџџџD
 
_user_specified_nameinputs

E
)__inference_dropout_1_layer_call_fn_23068

inputs
identityТ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_209662
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*&
_input_shapes
:џџџџџџџџџ:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
РU
ј	
E__inference_sequential_layer_call_and_return_conditional_losses_21087
reshape_input
melspectrogram_21011
batch_normalization_21014
batch_normalization_21016
batch_normalization_21018
batch_normalization_21020
conv2d_21023
conv2d_21025
batch_normalization_1_21029
batch_normalization_1_21031
batch_normalization_1_21033
batch_normalization_1_21035
conv2d_1_21038
conv2d_1_21040
batch_normalization_2_21044
batch_normalization_2_21046
batch_normalization_2_21048
batch_normalization_2_21050
conv2d_2_21053
conv2d_2_21055
batch_normalization_3_21059
batch_normalization_3_21061
batch_normalization_3_21063
batch_normalization_3_21065
dense_21069
dense_21071
dense_1_21075
dense_1_21077
dense_2_21081
dense_2_21083
identityЂ+batch_normalization/StatefulPartitionedCallЂ-batch_normalization_1/StatefulPartitionedCallЂ-batch_normalization_2/StatefulPartitionedCallЂ-batch_normalization_3/StatefulPartitionedCallЂconv2d/StatefulPartitionedCallЂ conv2d_1/StatefulPartitionedCallЂ conv2d_2/StatefulPartitionedCallЂdense/StatefulPartitionedCallЂdense_1/StatefulPartitionedCallЂdense_2/StatefulPartitionedCallн
reshape/PartitionedCallPartitionedCallreshape_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:џџџџџџџџџт	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_reshape_layer_call_and_return_conditional_losses_204502
reshape/PartitionedCall 
melspectrogram/PartitionedCallPartitionedCall reshape/PartitionedCall:output:0melspectrogram_21011*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:џџџџџџџџџЗ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_melspectrogram_layer_call_and_return_conditional_losses_199762 
melspectrogram/PartitionedCall­
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'melspectrogram/PartitionedCall:output:0batch_normalization_21014batch_normalization_21016batch_normalization_21018batch_normalization_21020*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:џџџџџџџџџЗ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_205122-
+batch_normalization/StatefulPartitionedCallП
conv2d/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0conv2d_21023conv2d_21025*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:џџџџџџџџџЗ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_205592 
conv2d/StatefulPartitionedCall
max_pooling2d/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџg@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_200912
max_pooling2d/PartitionedCallИ
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0batch_normalization_1_21029batch_normalization_1_21031batch_normalization_1_21033batch_normalization_1_21035*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџg@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_206132/
-batch_normalization_1/StatefulPartitionedCallЩ
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0conv2d_1_21038conv2d_1_21040*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџg@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_206602"
 conv2d_1/StatefulPartitionedCall
max_pooling2d_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ" * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_202072!
max_pooling2d_1/PartitionedCallК
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0batch_normalization_2_21044batch_normalization_2_21046batch_normalization_2_21048batch_normalization_2_21050*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ" *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_207142/
-batch_normalization_2/StatefulPartitionedCallЩ
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0conv2d_2_21053conv2d_2_21055*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ"  *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_conv2d_2_layer_call_and_return_conditional_losses_207612"
 conv2d_2/StatefulPartitionedCall
max_pooling2d_2/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_203232!
max_pooling2d_2/PartitionedCallК
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_2/PartitionedCall:output:0batch_normalization_3_21059batch_normalization_3_21061batch_normalization_3_21063batch_normalization_3_21065*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_208152/
-batch_normalization_3/StatefulPartitionedCall
flatten/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџD* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_208572
flatten/PartitionedCall
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_21069dense_21071*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_208762
dense/StatefulPartitionedCall№
dropout/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_209092
dropout/PartitionedCallІ
dense_1/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0dense_1_21075dense_1_21077*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_209332!
dense_1/StatefulPartitionedCallј
dropout_1/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_209662
dropout_1/PartitionedCallЈ
dense_2/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0dense_2_21081dense_2_21083*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_209902!
dense_2/StatefulPartitionedCall
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*І
_input_shapes
:џџџџџџџџџт	:
::::::::::::::::::::::::::::2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall:X T
)
_output_shapes
:џџџџџџџџџт	
'
_user_specified_namereshape_input:&"
 
_output_shapes
:

Э

м
C__inference_conv2d_1_layer_call_and_return_conditional_losses_22678

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpЃ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџg@*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџg@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџg@2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:џџџџџџџџџg@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџg@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџg@
 
_user_specified_nameinputs
њ
}
(__inference_conv2d_1_layer_call_fn_22687

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallћ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџg@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_206602
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:џџџџџџџџџg@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџg@::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџg@
 
_user_specified_nameinputs
ю	
л
B__inference_dense_2_layer_call_and_return_conditional_losses_20990

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
Sigmoid
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ь

P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_22855

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityЂAssignNewValueЂAssignNewValue_1ЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ : : : : :*
epsilon%o:*
exponential_avg_factor%
з#<2
FusedBatchNormV3­
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValueЛ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 
_user_specified_nameinputs
с

I__inference_melspectrogram_layer_call_and_return_conditional_losses_19960

inputs
apply_filterbank_19956
identityб
stft/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:џџџџџџџџџЗ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_stft_layer_call_and_return_conditional_losses_198772
stft/PartitionedCallї
magnitude/PartitionedCallPartitionedCallstft/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:џџџџџџџџџЗ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_magnitude_layer_call_and_return_conditional_losses_198902
magnitude/PartitionedCallЊ
 apply_filterbank/PartitionedCallPartitionedCall"magnitude/PartitionedCall:output:0apply_filterbank_19956*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:џџџџџџџџџЗ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_apply_filterbank_layer_call_and_return_conditional_losses_199262"
 apply_filterbank/PartitionedCall
IdentityIdentity)apply_filterbank/PartitionedCall:output:0*
T0*1
_output_shapes
:џџџџџџџџџЗ2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:џџџџџџџџџт	:
:U Q
-
_output_shapes
:џџџџџџџџџт	
 
_user_specified_nameinputs:&"
 
_output_shapes
:

Ь

P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_20275

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityЂAssignNewValueЂAssignNewValue_1ЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:*
exponential_avg_factor%
з#<2
FusedBatchNormV3­
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValueЛ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
О
ё
N__inference_batch_normalization_layer_call_and_return_conditional_losses_20074

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1м
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3ь
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
й
І
3__inference_batch_normalization_layer_call_fn_22519

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityЂStatefulPartitionedCallЂ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:џџџџџџџџџЗ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_205122
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:џџџџџџџџџЗ2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:џџџџџџџџџЗ::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:џџџџџџџџџЗ
 
_user_specified_nameinputs
ў

a
B__inference_dropout_layer_call_and_return_conditional_losses_23006

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeД
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>2
dropout/GreaterEqual/yО
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ@2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*&
_input_shapes
:џџџџџџџџџ@:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
з
І
3__inference_batch_normalization_layer_call_fn_22506

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityЂStatefulPartitionedCall 
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:џџџџџџџџџЗ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_204942
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:џџџџџџџџџЗ2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:џџџџџџџџџЗ::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:џџџџџџџџџЗ
 
_user_specified_nameinputs
ь	
л
B__inference_dense_1_layer_call_and_return_conditional_losses_23032

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs


P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_20595

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityЂAssignNewValueЂAssignNewValue_1ЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1и
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџg@:::::*
epsilon%o:*
exponential_avg_factor%
з#<2
FusedBatchNormV3­
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValueЛ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1ў
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:џџџџџџџџџg@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџg@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:џџџџџџџџџg@
 
_user_specified_nameinputs
Л
[
D__inference_magnitude_layer_call_and_return_conditional_losses_19890
x
identityN
Abs
ComplexAbsx*1
_output_shapes
:џџџџџџџџџЗ2
Abse
IdentityIdentityAbs:y:0*
T0*1
_output_shapes
:џџџџџџџџџЗ2

Identity"
identityIdentity:output:0*0
_input_shapes
:џџџџџџџџџЗ:T P
1
_output_shapes
:џџџџџџџџџЗ

_user_specified_namex
ў
{
&__inference_conv2d_layer_call_fn_22539

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallћ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:џџџџџџџџџЗ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_205592
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:џџџџџџџџџЗ2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:џџџџџџџџџЗ::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:џџџџџџџџџЗ
 
_user_specified_nameinputs
Ч
b
D__inference_dropout_1_layer_call_and_return_conditional_losses_20966

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:џџџџџџџџџ2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:џџџџџџџџџ:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ћ
K
/__inference_max_pooling2d_2_layer_call_fn_20329

inputs
identityы
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_203232
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs

І
*__inference_sequential_layer_call_fn_22028

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27
identityЂStatefulPartitionedCallн
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27*)
Tin"
 2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*6
_read_only_resource_inputs
	*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_211702
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*І
_input_shapes
:џџџџџџџџџт	:
::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
)
_output_shapes
:џџџџџџџџџт	
 
_user_specified_nameinputs:&"
 
_output_shapes
:

Ъ

N__inference_batch_normalization_layer_call_and_return_conditional_losses_20043

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityЂAssignNewValueЂAssignNewValue_1ЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:*
exponential_avg_factor%
з#<2
FusedBatchNormV3­
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValueЛ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
 
C
'__inference_flatten_layer_call_fn_22974

inputs
identityС
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџD* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_208572
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:џџџџџџџџџD2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ :W S
/
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
я
V
?__inference_stft_layer_call_and_return_conditional_losses_19877
x
identityu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permw
	transpose	Transposextranspose/perm:output:0*
T0*-
_output_shapes
:џџџџџџџџџт	2
	transpose
 stft_tf.signal.stft/frame_lengthConst*
_output_shapes
: *
dtype0*
value
B :2"
 stft_tf.signal.stft/frame_length
stft_tf.signal.stft/frame_stepConst*
_output_shapes
: *
dtype0*
value
B :2 
stft_tf.signal.stft/frame_step
stft_tf.signal.stft/fft_lengthConst*
_output_shapes
: *
dtype0*
value
B :2 
stft_tf.signal.stft/fft_length
stft_tf.signal.stft/frame/axisConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2 
stft_tf.signal.stft/frame/axis
stft_tf.signal.stft/frame/ShapeShapetranspose:y:0*
T0*
_output_shapes
:2!
stft_tf.signal.stft/frame/Shape
stft_tf.signal.stft/frame/RankConst*
_output_shapes
: *
dtype0*
value	B :2 
stft_tf.signal.stft/frame/Rank
%stft_tf.signal.stft/frame/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2'
%stft_tf.signal.stft/frame/range/start
%stft_tf.signal.stft/frame/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2'
%stft_tf.signal.stft/frame/range/delta№
stft_tf.signal.stft/frame/rangeRange.stft_tf.signal.stft/frame/range/start:output:0'stft_tf.signal.stft/frame/Rank:output:0.stft_tf.signal.stft/frame/range/delta:output:0*
_output_shapes
:2!
stft_tf.signal.stft/frame/rangeБ
-stft_tf.signal.stft/frame/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2/
-stft_tf.signal.stft/frame/strided_slice/stackЌ
/stft_tf.signal.stft/frame/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 21
/stft_tf.signal.stft/frame/strided_slice/stack_1Ќ
/stft_tf.signal.stft/frame/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/stft_tf.signal.stft/frame/strided_slice/stack_2ў
'stft_tf.signal.stft/frame/strided_sliceStridedSlice(stft_tf.signal.stft/frame/range:output:06stft_tf.signal.stft/frame/strided_slice/stack:output:08stft_tf.signal.stft/frame/strided_slice/stack_1:output:08stft_tf.signal.stft/frame/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2)
'stft_tf.signal.stft/frame/strided_slice
stft_tf.signal.stft/frame/sub/yConst*
_output_shapes
: *
dtype0*
value	B :2!
stft_tf.signal.stft/frame/sub/yЙ
stft_tf.signal.stft/frame/subSub'stft_tf.signal.stft/frame/Rank:output:0(stft_tf.signal.stft/frame/sub/y:output:0*
T0*
_output_shapes
: 2
stft_tf.signal.stft/frame/subП
stft_tf.signal.stft/frame/sub_1Sub!stft_tf.signal.stft/frame/sub:z:00stft_tf.signal.stft/frame/strided_slice:output:0*
T0*
_output_shapes
: 2!
stft_tf.signal.stft/frame/sub_1
"stft_tf.signal.stft/frame/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2$
"stft_tf.signal.stft/frame/packed/1ў
 stft_tf.signal.stft/frame/packedPack0stft_tf.signal.stft/frame/strided_slice:output:0+stft_tf.signal.stft/frame/packed/1:output:0#stft_tf.signal.stft/frame/sub_1:z:0*
N*
T0*
_output_shapes
:2"
 stft_tf.signal.stft/frame/packed
)stft_tf.signal.stft/frame/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2+
)stft_tf.signal.stft/frame/split/split_dimЁ
stft_tf.signal.stft/frame/splitSplitV(stft_tf.signal.stft/frame/Shape:output:0)stft_tf.signal.stft/frame/packed:output:02stft_tf.signal.stft/frame/split/split_dim:output:0*
T0*

Tlen0*$
_output_shapes
::: *
	num_split2!
stft_tf.signal.stft/frame/split
'stft_tf.signal.stft/frame/Reshape/shapeConst*
_output_shapes
: *
dtype0*
valueB 2)
'stft_tf.signal.stft/frame/Reshape/shape
)stft_tf.signal.stft/frame/Reshape/shape_1Const*
_output_shapes
: *
dtype0*
valueB 2+
)stft_tf.signal.stft/frame/Reshape/shape_1а
!stft_tf.signal.stft/frame/ReshapeReshape(stft_tf.signal.stft/frame/split:output:12stft_tf.signal.stft/frame/Reshape/shape_1:output:0*
T0*
_output_shapes
: 2#
!stft_tf.signal.stft/frame/Reshape
stft_tf.signal.stft/frame/SizeConst*
_output_shapes
: *
dtype0*
value	B :2 
stft_tf.signal.stft/frame/Size
 stft_tf.signal.stft/frame/Size_1Const*
_output_shapes
: *
dtype0*
value	B : 2"
 stft_tf.signal.stft/frame/Size_1С
stft_tf.signal.stft/frame/sub_2Sub*stft_tf.signal.stft/frame/Reshape:output:0)stft_tf.signal.stft/frame_length:output:0*
T0*
_output_shapes
: 2!
stft_tf.signal.stft/frame/sub_2У
"stft_tf.signal.stft/frame/floordivFloorDiv#stft_tf.signal.stft/frame/sub_2:z:0'stft_tf.signal.stft/frame_step:output:0*
T0*
_output_shapes
: 2$
"stft_tf.signal.stft/frame/floordiv
stft_tf.signal.stft/frame/add/xConst*
_output_shapes
: *
dtype0*
value	B :2!
stft_tf.signal.stft/frame/add/xК
stft_tf.signal.stft/frame/addAddV2(stft_tf.signal.stft/frame/add/x:output:0&stft_tf.signal.stft/frame/floordiv:z:0*
T0*
_output_shapes
: 2
stft_tf.signal.stft/frame/add
#stft_tf.signal.stft/frame/Maximum/xConst*
_output_shapes
: *
dtype0*
value	B : 2%
#stft_tf.signal.stft/frame/Maximum/xУ
!stft_tf.signal.stft/frame/MaximumMaximum,stft_tf.signal.stft/frame/Maximum/x:output:0!stft_tf.signal.stft/frame/add:z:0*
T0*
_output_shapes
: 2#
!stft_tf.signal.stft/frame/Maximum
#stft_tf.signal.stft/frame/gcd/ConstConst*
_output_shapes
: *
dtype0*
value
B :2%
#stft_tf.signal.stft/frame/gcd/Const
&stft_tf.signal.stft/frame/floordiv_1/yConst*
_output_shapes
: *
dtype0*
value
B :2(
&stft_tf.signal.stft/frame/floordiv_1/yе
$stft_tf.signal.stft/frame/floordiv_1FloorDiv)stft_tf.signal.stft/frame_length:output:0/stft_tf.signal.stft/frame/floordiv_1/y:output:0*
T0*
_output_shapes
: 2&
$stft_tf.signal.stft/frame/floordiv_1
&stft_tf.signal.stft/frame/floordiv_2/yConst*
_output_shapes
: *
dtype0*
value
B :2(
&stft_tf.signal.stft/frame/floordiv_2/yг
$stft_tf.signal.stft/frame/floordiv_2FloorDiv'stft_tf.signal.stft/frame_step:output:0/stft_tf.signal.stft/frame/floordiv_2/y:output:0*
T0*
_output_shapes
: 2&
$stft_tf.signal.stft/frame/floordiv_2
&stft_tf.signal.stft/frame/floordiv_3/yConst*
_output_shapes
: *
dtype0*
value
B :2(
&stft_tf.signal.stft/frame/floordiv_3/yж
$stft_tf.signal.stft/frame/floordiv_3FloorDiv*stft_tf.signal.stft/frame/Reshape:output:0/stft_tf.signal.stft/frame/floordiv_3/y:output:0*
T0*
_output_shapes
: 2&
$stft_tf.signal.stft/frame/floordiv_3
stft_tf.signal.stft/frame/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2!
stft_tf.signal.stft/frame/mul/yК
stft_tf.signal.stft/frame/mulMul(stft_tf.signal.stft/frame/floordiv_3:z:0(stft_tf.signal.stft/frame/mul/y:output:0*
T0*
_output_shapes
: 2
stft_tf.signal.stft/frame/mulЏ
)stft_tf.signal.stft/frame/concat/values_1Pack!stft_tf.signal.stft/frame/mul:z:0*
N*
T0*
_output_shapes
:2+
)stft_tf.signal.stft/frame/concat/values_1
%stft_tf.signal.stft/frame/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2'
%stft_tf.signal.stft/frame/concat/axisЖ
 stft_tf.signal.stft/frame/concatConcatV2(stft_tf.signal.stft/frame/split:output:02stft_tf.signal.stft/frame/concat/values_1:output:0(stft_tf.signal.stft/frame/split:output:2.stft_tf.signal.stft/frame/concat/axis:output:0*
N*
T0*
_output_shapes
:2"
 stft_tf.signal.stft/frame/concatЁ
-stft_tf.signal.stft/frame/concat_1/values_1/1Const*
_output_shapes
: *
dtype0*
value
B :2/
-stft_tf.signal.stft/frame/concat_1/values_1/1ђ
+stft_tf.signal.stft/frame/concat_1/values_1Pack(stft_tf.signal.stft/frame/floordiv_3:z:06stft_tf.signal.stft/frame/concat_1/values_1/1:output:0*
N*
T0*
_output_shapes
:2-
+stft_tf.signal.stft/frame/concat_1/values_1
'stft_tf.signal.stft/frame/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2)
'stft_tf.signal.stft/frame/concat_1/axisО
"stft_tf.signal.stft/frame/concat_1ConcatV2(stft_tf.signal.stft/frame/split:output:04stft_tf.signal.stft/frame/concat_1/values_1:output:0(stft_tf.signal.stft/frame/split:output:20stft_tf.signal.stft/frame/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2$
"stft_tf.signal.stft/frame/concat_1
$stft_tf.signal.stft/frame/zeros_likeConst*
_output_shapes
:*
dtype0*
valueB: 2&
$stft_tf.signal.stft/frame/zeros_like 
)stft_tf.signal.stft/frame/ones_like/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2+
)stft_tf.signal.stft/frame/ones_like/Shape
)stft_tf.signal.stft/frame/ones_like/ConstConst*
_output_shapes
: *
dtype0*
value	B :2+
)stft_tf.signal.stft/frame/ones_like/Constп
#stft_tf.signal.stft/frame/ones_likeFill2stft_tf.signal.stft/frame/ones_like/Shape:output:02stft_tf.signal.stft/frame/ones_like/Const:output:0*
T0*
_output_shapes
:2%
#stft_tf.signal.stft/frame/ones_likeЬ
&stft_tf.signal.stft/frame/StridedSliceStridedSlicetranspose:y:0-stft_tf.signal.stft/frame/zeros_like:output:0)stft_tf.signal.stft/frame/concat:output:0,stft_tf.signal.stft/frame/ones_like:output:0*
Index0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ2(
&stft_tf.signal.stft/frame/StridedSlice
#stft_tf.signal.stft/frame/Reshape_1Reshape/stft_tf.signal.stft/frame/StridedSlice:output:0+stft_tf.signal.stft/frame/concat_1:output:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2%
#stft_tf.signal.stft/frame/Reshape_1
'stft_tf.signal.stft/frame/range_1/startConst*
_output_shapes
: *
dtype0*
value	B : 2)
'stft_tf.signal.stft/frame/range_1/start
'stft_tf.signal.stft/frame/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :2)
'stft_tf.signal.stft/frame/range_1/deltaџ
!stft_tf.signal.stft/frame/range_1Range0stft_tf.signal.stft/frame/range_1/start:output:0%stft_tf.signal.stft/frame/Maximum:z:00stft_tf.signal.stft/frame/range_1/delta:output:0*#
_output_shapes
:џџџџџџџџџ2#
!stft_tf.signal.stft/frame/range_1Э
stft_tf.signal.stft/frame/mul_1Mul*stft_tf.signal.stft/frame/range_1:output:0(stft_tf.signal.stft/frame/floordiv_2:z:0*
T0*#
_output_shapes
:џџџџџџџџџ2!
stft_tf.signal.stft/frame/mul_1
+stft_tf.signal.stft/frame/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2-
+stft_tf.signal.stft/frame/Reshape_2/shape/1щ
)stft_tf.signal.stft/frame/Reshape_2/shapePack%stft_tf.signal.stft/frame/Maximum:z:04stft_tf.signal.stft/frame/Reshape_2/shape/1:output:0*
N*
T0*
_output_shapes
:2+
)stft_tf.signal.stft/frame/Reshape_2/shapeр
#stft_tf.signal.stft/frame/Reshape_2Reshape#stft_tf.signal.stft/frame/mul_1:z:02stft_tf.signal.stft/frame/Reshape_2/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2%
#stft_tf.signal.stft/frame/Reshape_2
'stft_tf.signal.stft/frame/range_2/startConst*
_output_shapes
: *
dtype0*
value	B : 2)
'stft_tf.signal.stft/frame/range_2/start
'stft_tf.signal.stft/frame/range_2/deltaConst*
_output_shapes
: *
dtype0*
value	B :2)
'stft_tf.signal.stft/frame/range_2/deltaљ
!stft_tf.signal.stft/frame/range_2Range0stft_tf.signal.stft/frame/range_2/start:output:0(stft_tf.signal.stft/frame/floordiv_1:z:00stft_tf.signal.stft/frame/range_2/delta:output:0*
_output_shapes
:2#
!stft_tf.signal.stft/frame/range_2
+stft_tf.signal.stft/frame/Reshape_3/shape/0Const*
_output_shapes
: *
dtype0*
value	B :2-
+stft_tf.signal.stft/frame/Reshape_3/shape/0ь
)stft_tf.signal.stft/frame/Reshape_3/shapePack4stft_tf.signal.stft/frame/Reshape_3/shape/0:output:0(stft_tf.signal.stft/frame/floordiv_1:z:0*
N*
T0*
_output_shapes
:2+
)stft_tf.signal.stft/frame/Reshape_3/shapeо
#stft_tf.signal.stft/frame/Reshape_3Reshape*stft_tf.signal.stft/frame/range_2:output:02stft_tf.signal.stft/frame/Reshape_3/shape:output:0*
T0*
_output_shapes

:2%
#stft_tf.signal.stft/frame/Reshape_3й
stft_tf.signal.stft/frame/add_1AddV2,stft_tf.signal.stft/frame/Reshape_2:output:0,stft_tf.signal.stft/frame/Reshape_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2!
stft_tf.signal.stft/frame/add_1Э
"stft_tf.signal.stft/frame/GatherV2GatherV2,stft_tf.signal.stft/frame/Reshape_1:output:0#stft_tf.signal.stft/frame/add_1:z:00stft_tf.signal.stft/frame/strided_slice:output:0*
Taxis0*
Tindices0*
Tparams0*F
_output_shapes4
2:0џџџџџџџџџџџџџџџџџџџџџџџџџџџ2$
"stft_tf.signal.stft/frame/GatherV2т
+stft_tf.signal.stft/frame/concat_2/values_1Pack%stft_tf.signal.stft/frame/Maximum:z:0)stft_tf.signal.stft/frame_length:output:0*
N*
T0*
_output_shapes
:2-
+stft_tf.signal.stft/frame/concat_2/values_1
'stft_tf.signal.stft/frame/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2)
'stft_tf.signal.stft/frame/concat_2/axisО
"stft_tf.signal.stft/frame/concat_2ConcatV2(stft_tf.signal.stft/frame/split:output:04stft_tf.signal.stft/frame/concat_2/values_1:output:0(stft_tf.signal.stft/frame/split:output:20stft_tf.signal.stft/frame/concat_2/axis:output:0*
N*
T0*
_output_shapes
:2$
"stft_tf.signal.stft/frame/concat_2ы
#stft_tf.signal.stft/frame/Reshape_4Reshape+stft_tf.signal.stft/frame/GatherV2:output:0+stft_tf.signal.stft/frame/concat_2:output:0*
T0*1
_output_shapes
:џџџџџџџџџЗ2%
#stft_tf.signal.stft/frame/Reshape_4
(stft_tf.signal.stft/hann_window/periodicConst*
_output_shapes
: *
dtype0
*
value	B
 Z2*
(stft_tf.signal.stft/hann_window/periodicЗ
$stft_tf.signal.stft/hann_window/CastCast1stft_tf.signal.stft/hann_window/periodic:output:0*

DstT0*

SrcT0
*
_output_shapes
: 2&
$stft_tf.signal.stft/hann_window/Cast
*stft_tf.signal.stft/hann_window/FloorMod/yConst*
_output_shapes
: *
dtype0*
value	B :2,
*stft_tf.signal.stft/hann_window/FloorMod/yс
(stft_tf.signal.stft/hann_window/FloorModFloorMod)stft_tf.signal.stft/frame_length:output:03stft_tf.signal.stft/hann_window/FloorMod/y:output:0*
T0*
_output_shapes
: 2*
(stft_tf.signal.stft/hann_window/FloorMod
%stft_tf.signal.stft/hann_window/sub/xConst*
_output_shapes
: *
dtype0*
value	B :2'
%stft_tf.signal.stft/hann_window/sub/xа
#stft_tf.signal.stft/hann_window/subSub.stft_tf.signal.stft/hann_window/sub/x:output:0,stft_tf.signal.stft/hann_window/FloorMod:z:0*
T0*
_output_shapes
: 2%
#stft_tf.signal.stft/hann_window/subХ
#stft_tf.signal.stft/hann_window/mulMul(stft_tf.signal.stft/hann_window/Cast:y:0'stft_tf.signal.stft/hann_window/sub:z:0*
T0*
_output_shapes
: 2%
#stft_tf.signal.stft/hann_window/mulШ
#stft_tf.signal.stft/hann_window/addAddV2)stft_tf.signal.stft/frame_length:output:0'stft_tf.signal.stft/hann_window/mul:z:0*
T0*
_output_shapes
: 2%
#stft_tf.signal.stft/hann_window/add
'stft_tf.signal.stft/hann_window/sub_1/yConst*
_output_shapes
: *
dtype0*
value	B :2)
'stft_tf.signal.stft/hann_window/sub_1/yб
%stft_tf.signal.stft/hann_window/sub_1Sub'stft_tf.signal.stft/hann_window/add:z:00stft_tf.signal.stft/hann_window/sub_1/y:output:0*
T0*
_output_shapes
: 2'
%stft_tf.signal.stft/hann_window/sub_1Г
&stft_tf.signal.stft/hann_window/Cast_1Cast)stft_tf.signal.stft/hann_window/sub_1:z:0*

DstT0*

SrcT0*
_output_shapes
: 2(
&stft_tf.signal.stft/hann_window/Cast_1
+stft_tf.signal.stft/hann_window/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2-
+stft_tf.signal.stft/hann_window/range/start
+stft_tf.signal.stft/hann_window/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2-
+stft_tf.signal.stft/hann_window/range/delta
%stft_tf.signal.stft/hann_window/rangeRange4stft_tf.signal.stft/hann_window/range/start:output:0)stft_tf.signal.stft/frame_length:output:04stft_tf.signal.stft/hann_window/range/delta:output:0*
_output_shapes	
:2'
%stft_tf.signal.stft/hann_window/rangeН
&stft_tf.signal.stft/hann_window/Cast_2Cast.stft_tf.signal.stft/hann_window/range:output:0*

DstT0*

SrcT0*
_output_shapes	
:2(
&stft_tf.signal.stft/hann_window/Cast_2
%stft_tf.signal.stft/hann_window/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *лЩ@2'
%stft_tf.signal.stft/hann_window/Constз
%stft_tf.signal.stft/hann_window/mul_1Mul.stft_tf.signal.stft/hann_window/Const:output:0*stft_tf.signal.stft/hann_window/Cast_2:y:0*
T0*
_output_shapes	
:2'
%stft_tf.signal.stft/hann_window/mul_1к
'stft_tf.signal.stft/hann_window/truedivRealDiv)stft_tf.signal.stft/hann_window/mul_1:z:0*stft_tf.signal.stft/hann_window/Cast_1:y:0*
T0*
_output_shapes	
:2)
'stft_tf.signal.stft/hann_window/truedivЄ
#stft_tf.signal.stft/hann_window/CosCos+stft_tf.signal.stft/hann_window/truediv:z:0*
T0*
_output_shapes	
:2%
#stft_tf.signal.stft/hann_window/Cos
'stft_tf.signal.stft/hann_window/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2)
'stft_tf.signal.stft/hann_window/mul_2/xж
%stft_tf.signal.stft/hann_window/mul_2Mul0stft_tf.signal.stft/hann_window/mul_2/x:output:0'stft_tf.signal.stft/hann_window/Cos:y:0*
T0*
_output_shapes	
:2'
%stft_tf.signal.stft/hann_window/mul_2
'stft_tf.signal.stft/hann_window/sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2)
'stft_tf.signal.stft/hann_window/sub_2/xи
%stft_tf.signal.stft/hann_window/sub_2Sub0stft_tf.signal.stft/hann_window/sub_2/x:output:0)stft_tf.signal.stft/hann_window/mul_2:z:0*
T0*
_output_shapes	
:2'
%stft_tf.signal.stft/hann_window/sub_2Ю
stft_tf.signal.stft/mulMul,stft_tf.signal.stft/frame/Reshape_4:output:0)stft_tf.signal.stft/hann_window/sub_2:z:0*
T0*1
_output_shapes
:џџџџџџџџџЗ2
stft_tf.signal.stft/mulЁ
stft_tf.signal.stft/rfft/packedPack'stft_tf.signal.stft/fft_length:output:0*
N*
T0*
_output_shapes
:2!
stft_tf.signal.stft/rfft/packed
#stft_tf.signal.stft/rfft/fft_lengthConst*
_output_shapes
:*
dtype0*
valueB:2%
#stft_tf.signal.stft/rfft/fft_lengthК
stft_tf.signal.stft/rfftRFFTstft_tf.signal.stft/mul:z:0,stft_tf.signal.stft/rfft/fft_length:output:0*1
_output_shapes
:џџџџџџџџџЗ2
stft_tf.signal.stft/rfft}
transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             2
transpose_1/permЁ
transpose_1	Transpose!stft_tf.signal.stft/rfft:output:0transpose_1/perm:output:0*
T0*1
_output_shapes
:џџџџџџџџџЗ2
transpose_1m
IdentityIdentitytranspose_1:y:0*
T0*1
_output_shapes
:џџџџџџџџџЗ2

Identity"
identityIdentity:output:0*,
_input_shapes
:џџџџџџџџџт	:P L
-
_output_shapes
:џџџџџџџџџт	

_user_specified_namex

W
.__inference_melspectrogram_layer_call_fn_22391

inputs
unknown
identityл
PartitionedCallPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:џџџџџџџџџЗ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_melspectrogram_layer_call_and_return_conditional_losses_199762
PartitionedCallv
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:џџџџџџџџџЗ2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:џџџџџџџџџт	:
:U Q
-
_output_shapes
:џџџџџџџџџт	
 
_user_specified_nameinputs:&"
 
_output_shapes
:

Р
ѓ
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_22873

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1м
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ : : : : :*
epsilon%o:*
is_training( 2
FusedBatchNormV3ь
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 
_user_specified_nameinputs
Р
ѓ
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_20306

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1м
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3ь
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs

b
)__inference_dropout_1_layer_call_fn_23063

inputs
identityЂStatefulPartitionedCallк
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_209612
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*&
_input_shapes
:џџџџџџџџџ22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ј
ѓ
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_20613

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџg@:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3к
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:џџџџџџџџџg@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџg@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:џџџџџџџџџg@
 
_user_specified_nameinputs
в
Т
 __inference__wrapped_model_19764
reshape_input:
6sequential_melspectrogram_apply_filterbank_tensordot_b:
6sequential_batch_normalization_readvariableop_resource<
8sequential_batch_normalization_readvariableop_1_resourceK
Gsequential_batch_normalization_fusedbatchnormv3_readvariableop_resourceM
Isequential_batch_normalization_fusedbatchnormv3_readvariableop_1_resource4
0sequential_conv2d_conv2d_readvariableop_resource5
1sequential_conv2d_biasadd_readvariableop_resource<
8sequential_batch_normalization_1_readvariableop_resource>
:sequential_batch_normalization_1_readvariableop_1_resourceM
Isequential_batch_normalization_1_fusedbatchnormv3_readvariableop_resourceO
Ksequential_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource6
2sequential_conv2d_1_conv2d_readvariableop_resource7
3sequential_conv2d_1_biasadd_readvariableop_resource<
8sequential_batch_normalization_2_readvariableop_resource>
:sequential_batch_normalization_2_readvariableop_1_resourceM
Isequential_batch_normalization_2_fusedbatchnormv3_readvariableop_resourceO
Ksequential_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource6
2sequential_conv2d_2_conv2d_readvariableop_resource7
3sequential_conv2d_2_biasadd_readvariableop_resource<
8sequential_batch_normalization_3_readvariableop_resource>
:sequential_batch_normalization_3_readvariableop_1_resourceM
Isequential_batch_normalization_3_fusedbatchnormv3_readvariableop_resourceO
Ksequential_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource3
/sequential_dense_matmul_readvariableop_resource4
0sequential_dense_biasadd_readvariableop_resource5
1sequential_dense_1_matmul_readvariableop_resource6
2sequential_dense_1_biasadd_readvariableop_resource5
1sequential_dense_2_matmul_readvariableop_resource6
2sequential_dense_2_biasadd_readvariableop_resource
identityЂ>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOpЂ@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1Ђ-sequential/batch_normalization/ReadVariableOpЂ/sequential/batch_normalization/ReadVariableOp_1Ђ@sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOpЂBsequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1Ђ/sequential/batch_normalization_1/ReadVariableOpЂ1sequential/batch_normalization_1/ReadVariableOp_1Ђ@sequential/batch_normalization_2/FusedBatchNormV3/ReadVariableOpЂBsequential/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1Ђ/sequential/batch_normalization_2/ReadVariableOpЂ1sequential/batch_normalization_2/ReadVariableOp_1Ђ@sequential/batch_normalization_3/FusedBatchNormV3/ReadVariableOpЂBsequential/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1Ђ/sequential/batch_normalization_3/ReadVariableOpЂ1sequential/batch_normalization_3/ReadVariableOp_1Ђ(sequential/conv2d/BiasAdd/ReadVariableOpЂ'sequential/conv2d/Conv2D/ReadVariableOpЂ*sequential/conv2d_1/BiasAdd/ReadVariableOpЂ)sequential/conv2d_1/Conv2D/ReadVariableOpЂ*sequential/conv2d_2/BiasAdd/ReadVariableOpЂ)sequential/conv2d_2/Conv2D/ReadVariableOpЂ'sequential/dense/BiasAdd/ReadVariableOpЂ&sequential/dense/MatMul/ReadVariableOpЂ)sequential/dense_1/BiasAdd/ReadVariableOpЂ(sequential/dense_1/MatMul/ReadVariableOpЂ)sequential/dense_2/BiasAdd/ReadVariableOpЂ(sequential/dense_2/MatMul/ReadVariableOpq
sequential/reshape/ShapeShapereshape_input*
T0*
_output_shapes
:2
sequential/reshape/Shape
&sequential/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential/reshape/strided_slice/stack
(sequential/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(sequential/reshape/strided_slice/stack_1
(sequential/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(sequential/reshape/strided_slice/stack_2д
 sequential/reshape/strided_sliceStridedSlice!sequential/reshape/Shape:output:0/sequential/reshape/strided_slice/stack:output:01sequential/reshape/strided_slice/stack_1:output:01sequential/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 sequential/reshape/strided_slice
"sequential/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB	 :т	2$
"sequential/reshape/Reshape/shape/1
"sequential/reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2$
"sequential/reshape/Reshape/shape/2џ
 sequential/reshape/Reshape/shapePack)sequential/reshape/strided_slice:output:0+sequential/reshape/Reshape/shape/1:output:0+sequential/reshape/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2"
 sequential/reshape/Reshape/shapeЕ
sequential/reshape/ReshapeReshapereshape_input)sequential/reshape/Reshape/shape:output:0*
T0*-
_output_shapes
:џџџџџџџџџт	2
sequential/reshape/ReshapeГ
-sequential/melspectrogram/stft/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2/
-sequential/melspectrogram/stft/transpose/permі
(sequential/melspectrogram/stft/transpose	Transpose#sequential/reshape/Reshape:output:06sequential/melspectrogram/stft/transpose/perm:output:0*
T0*-
_output_shapes
:џџџџџџџџџт	2*
(sequential/melspectrogram/stft/transposeХ
?sequential/melspectrogram/stft/stft_tf.signal.stft/frame_lengthConst*
_output_shapes
: *
dtype0*
value
B :2A
?sequential/melspectrogram/stft/stft_tf.signal.stft/frame_lengthС
=sequential/melspectrogram/stft/stft_tf.signal.stft/frame_stepConst*
_output_shapes
: *
dtype0*
value
B :2?
=sequential/melspectrogram/stft/stft_tf.signal.stft/frame_stepС
=sequential/melspectrogram/stft/stft_tf.signal.stft/fft_lengthConst*
_output_shapes
: *
dtype0*
value
B :2?
=sequential/melspectrogram/stft/stft_tf.signal.stft/fft_lengthЩ
=sequential/melspectrogram/stft/stft_tf.signal.stft/frame/axisConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2?
=sequential/melspectrogram/stft/stft_tf.signal.stft/frame/axisм
>sequential/melspectrogram/stft/stft_tf.signal.stft/frame/ShapeShape,sequential/melspectrogram/stft/transpose:y:0*
T0*
_output_shapes
:2@
>sequential/melspectrogram/stft/stft_tf.signal.stft/frame/ShapeР
=sequential/melspectrogram/stft/stft_tf.signal.stft/frame/RankConst*
_output_shapes
: *
dtype0*
value	B :2?
=sequential/melspectrogram/stft/stft_tf.signal.stft/frame/RankЮ
Dsequential/melspectrogram/stft/stft_tf.signal.stft/frame/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2F
Dsequential/melspectrogram/stft/stft_tf.signal.stft/frame/range/startЮ
Dsequential/melspectrogram/stft/stft_tf.signal.stft/frame/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2F
Dsequential/melspectrogram/stft/stft_tf.signal.stft/frame/range/delta
>sequential/melspectrogram/stft/stft_tf.signal.stft/frame/rangeRangeMsequential/melspectrogram/stft/stft_tf.signal.stft/frame/range/start:output:0Fsequential/melspectrogram/stft/stft_tf.signal.stft/frame/Rank:output:0Msequential/melspectrogram/stft/stft_tf.signal.stft/frame/range/delta:output:0*
_output_shapes
:2@
>sequential/melspectrogram/stft/stft_tf.signal.stft/frame/rangeя
Lsequential/melspectrogram/stft/stft_tf.signal.stft/frame/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2N
Lsequential/melspectrogram/stft/stft_tf.signal.stft/frame/strided_slice/stackъ
Nsequential/melspectrogram/stft/stft_tf.signal.stft/frame/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2P
Nsequential/melspectrogram/stft/stft_tf.signal.stft/frame/strided_slice/stack_1ъ
Nsequential/melspectrogram/stft/stft_tf.signal.stft/frame/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2P
Nsequential/melspectrogram/stft/stft_tf.signal.stft/frame/strided_slice/stack_2И
Fsequential/melspectrogram/stft/stft_tf.signal.stft/frame/strided_sliceStridedSliceGsequential/melspectrogram/stft/stft_tf.signal.stft/frame/range:output:0Usequential/melspectrogram/stft/stft_tf.signal.stft/frame/strided_slice/stack:output:0Wsequential/melspectrogram/stft/stft_tf.signal.stft/frame/strided_slice/stack_1:output:0Wsequential/melspectrogram/stft/stft_tf.signal.stft/frame/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2H
Fsequential/melspectrogram/stft/stft_tf.signal.stft/frame/strided_sliceТ
>sequential/melspectrogram/stft/stft_tf.signal.stft/frame/sub/yConst*
_output_shapes
: *
dtype0*
value	B :2@
>sequential/melspectrogram/stft/stft_tf.signal.stft/frame/sub/yЕ
<sequential/melspectrogram/stft/stft_tf.signal.stft/frame/subSubFsequential/melspectrogram/stft/stft_tf.signal.stft/frame/Rank:output:0Gsequential/melspectrogram/stft/stft_tf.signal.stft/frame/sub/y:output:0*
T0*
_output_shapes
: 2>
<sequential/melspectrogram/stft/stft_tf.signal.stft/frame/subЛ
>sequential/melspectrogram/stft/stft_tf.signal.stft/frame/sub_1Sub@sequential/melspectrogram/stft/stft_tf.signal.stft/frame/sub:z:0Osequential/melspectrogram/stft/stft_tf.signal.stft/frame/strided_slice:output:0*
T0*
_output_shapes
: 2@
>sequential/melspectrogram/stft/stft_tf.signal.stft/frame/sub_1Ш
Asequential/melspectrogram/stft/stft_tf.signal.stft/frame/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2C
Asequential/melspectrogram/stft/stft_tf.signal.stft/frame/packed/1
?sequential/melspectrogram/stft/stft_tf.signal.stft/frame/packedPackOsequential/melspectrogram/stft/stft_tf.signal.stft/frame/strided_slice:output:0Jsequential/melspectrogram/stft/stft_tf.signal.stft/frame/packed/1:output:0Bsequential/melspectrogram/stft/stft_tf.signal.stft/frame/sub_1:z:0*
N*
T0*
_output_shapes
:2A
?sequential/melspectrogram/stft/stft_tf.signal.stft/frame/packedж
Hsequential/melspectrogram/stft/stft_tf.signal.stft/frame/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2J
Hsequential/melspectrogram/stft/stft_tf.signal.stft/frame/split/split_dimМ
>sequential/melspectrogram/stft/stft_tf.signal.stft/frame/splitSplitVGsequential/melspectrogram/stft/stft_tf.signal.stft/frame/Shape:output:0Hsequential/melspectrogram/stft/stft_tf.signal.stft/frame/packed:output:0Qsequential/melspectrogram/stft/stft_tf.signal.stft/frame/split/split_dim:output:0*
T0*

Tlen0*$
_output_shapes
::: *
	num_split2@
>sequential/melspectrogram/stft/stft_tf.signal.stft/frame/splitг
Fsequential/melspectrogram/stft/stft_tf.signal.stft/frame/Reshape/shapeConst*
_output_shapes
: *
dtype0*
valueB 2H
Fsequential/melspectrogram/stft/stft_tf.signal.stft/frame/Reshape/shapeз
Hsequential/melspectrogram/stft/stft_tf.signal.stft/frame/Reshape/shape_1Const*
_output_shapes
: *
dtype0*
valueB 2J
Hsequential/melspectrogram/stft/stft_tf.signal.stft/frame/Reshape/shape_1Ь
@sequential/melspectrogram/stft/stft_tf.signal.stft/frame/ReshapeReshapeGsequential/melspectrogram/stft/stft_tf.signal.stft/frame/split:output:1Qsequential/melspectrogram/stft/stft_tf.signal.stft/frame/Reshape/shape_1:output:0*
T0*
_output_shapes
: 2B
@sequential/melspectrogram/stft/stft_tf.signal.stft/frame/ReshapeР
=sequential/melspectrogram/stft/stft_tf.signal.stft/frame/SizeConst*
_output_shapes
: *
dtype0*
value	B :2?
=sequential/melspectrogram/stft/stft_tf.signal.stft/frame/SizeФ
?sequential/melspectrogram/stft/stft_tf.signal.stft/frame/Size_1Const*
_output_shapes
: *
dtype0*
value	B : 2A
?sequential/melspectrogram/stft/stft_tf.signal.stft/frame/Size_1Н
>sequential/melspectrogram/stft/stft_tf.signal.stft/frame/sub_2SubIsequential/melspectrogram/stft/stft_tf.signal.stft/frame/Reshape:output:0Hsequential/melspectrogram/stft/stft_tf.signal.stft/frame_length:output:0*
T0*
_output_shapes
: 2@
>sequential/melspectrogram/stft/stft_tf.signal.stft/frame/sub_2П
Asequential/melspectrogram/stft/stft_tf.signal.stft/frame/floordivFloorDivBsequential/melspectrogram/stft/stft_tf.signal.stft/frame/sub_2:z:0Fsequential/melspectrogram/stft/stft_tf.signal.stft/frame_step:output:0*
T0*
_output_shapes
: 2C
Asequential/melspectrogram/stft/stft_tf.signal.stft/frame/floordivТ
>sequential/melspectrogram/stft/stft_tf.signal.stft/frame/add/xConst*
_output_shapes
: *
dtype0*
value	B :2@
>sequential/melspectrogram/stft/stft_tf.signal.stft/frame/add/xЖ
<sequential/melspectrogram/stft/stft_tf.signal.stft/frame/addAddV2Gsequential/melspectrogram/stft/stft_tf.signal.stft/frame/add/x:output:0Esequential/melspectrogram/stft/stft_tf.signal.stft/frame/floordiv:z:0*
T0*
_output_shapes
: 2>
<sequential/melspectrogram/stft/stft_tf.signal.stft/frame/addЪ
Bsequential/melspectrogram/stft/stft_tf.signal.stft/frame/Maximum/xConst*
_output_shapes
: *
dtype0*
value	B : 2D
Bsequential/melspectrogram/stft/stft_tf.signal.stft/frame/Maximum/xП
@sequential/melspectrogram/stft/stft_tf.signal.stft/frame/MaximumMaximumKsequential/melspectrogram/stft/stft_tf.signal.stft/frame/Maximum/x:output:0@sequential/melspectrogram/stft/stft_tf.signal.stft/frame/add:z:0*
T0*
_output_shapes
: 2B
@sequential/melspectrogram/stft/stft_tf.signal.stft/frame/MaximumЫ
Bsequential/melspectrogram/stft/stft_tf.signal.stft/frame/gcd/ConstConst*
_output_shapes
: *
dtype0*
value
B :2D
Bsequential/melspectrogram/stft/stft_tf.signal.stft/frame/gcd/Constб
Esequential/melspectrogram/stft/stft_tf.signal.stft/frame/floordiv_1/yConst*
_output_shapes
: *
dtype0*
value
B :2G
Esequential/melspectrogram/stft/stft_tf.signal.stft/frame/floordiv_1/yб
Csequential/melspectrogram/stft/stft_tf.signal.stft/frame/floordiv_1FloorDivHsequential/melspectrogram/stft/stft_tf.signal.stft/frame_length:output:0Nsequential/melspectrogram/stft/stft_tf.signal.stft/frame/floordiv_1/y:output:0*
T0*
_output_shapes
: 2E
Csequential/melspectrogram/stft/stft_tf.signal.stft/frame/floordiv_1б
Esequential/melspectrogram/stft/stft_tf.signal.stft/frame/floordiv_2/yConst*
_output_shapes
: *
dtype0*
value
B :2G
Esequential/melspectrogram/stft/stft_tf.signal.stft/frame/floordiv_2/yЯ
Csequential/melspectrogram/stft/stft_tf.signal.stft/frame/floordiv_2FloorDivFsequential/melspectrogram/stft/stft_tf.signal.stft/frame_step:output:0Nsequential/melspectrogram/stft/stft_tf.signal.stft/frame/floordiv_2/y:output:0*
T0*
_output_shapes
: 2E
Csequential/melspectrogram/stft/stft_tf.signal.stft/frame/floordiv_2б
Esequential/melspectrogram/stft/stft_tf.signal.stft/frame/floordiv_3/yConst*
_output_shapes
: *
dtype0*
value
B :2G
Esequential/melspectrogram/stft/stft_tf.signal.stft/frame/floordiv_3/yв
Csequential/melspectrogram/stft/stft_tf.signal.stft/frame/floordiv_3FloorDivIsequential/melspectrogram/stft/stft_tf.signal.stft/frame/Reshape:output:0Nsequential/melspectrogram/stft/stft_tf.signal.stft/frame/floordiv_3/y:output:0*
T0*
_output_shapes
: 2E
Csequential/melspectrogram/stft/stft_tf.signal.stft/frame/floordiv_3У
>sequential/melspectrogram/stft/stft_tf.signal.stft/frame/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2@
>sequential/melspectrogram/stft/stft_tf.signal.stft/frame/mul/yЖ
<sequential/melspectrogram/stft/stft_tf.signal.stft/frame/mulMulGsequential/melspectrogram/stft/stft_tf.signal.stft/frame/floordiv_3:z:0Gsequential/melspectrogram/stft/stft_tf.signal.stft/frame/mul/y:output:0*
T0*
_output_shapes
: 2>
<sequential/melspectrogram/stft/stft_tf.signal.stft/frame/mul
Hsequential/melspectrogram/stft/stft_tf.signal.stft/frame/concat/values_1Pack@sequential/melspectrogram/stft/stft_tf.signal.stft/frame/mul:z:0*
N*
T0*
_output_shapes
:2J
Hsequential/melspectrogram/stft/stft_tf.signal.stft/frame/concat/values_1Ю
Dsequential/melspectrogram/stft/stft_tf.signal.stft/frame/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2F
Dsequential/melspectrogram/stft/stft_tf.signal.stft/frame/concat/axis№
?sequential/melspectrogram/stft/stft_tf.signal.stft/frame/concatConcatV2Gsequential/melspectrogram/stft/stft_tf.signal.stft/frame/split:output:0Qsequential/melspectrogram/stft/stft_tf.signal.stft/frame/concat/values_1:output:0Gsequential/melspectrogram/stft/stft_tf.signal.stft/frame/split:output:2Msequential/melspectrogram/stft/stft_tf.signal.stft/frame/concat/axis:output:0*
N*
T0*
_output_shapes
:2A
?sequential/melspectrogram/stft/stft_tf.signal.stft/frame/concatп
Lsequential/melspectrogram/stft/stft_tf.signal.stft/frame/concat_1/values_1/1Const*
_output_shapes
: *
dtype0*
value
B :2N
Lsequential/melspectrogram/stft/stft_tf.signal.stft/frame/concat_1/values_1/1ю
Jsequential/melspectrogram/stft/stft_tf.signal.stft/frame/concat_1/values_1PackGsequential/melspectrogram/stft/stft_tf.signal.stft/frame/floordiv_3:z:0Usequential/melspectrogram/stft/stft_tf.signal.stft/frame/concat_1/values_1/1:output:0*
N*
T0*
_output_shapes
:2L
Jsequential/melspectrogram/stft/stft_tf.signal.stft/frame/concat_1/values_1в
Fsequential/melspectrogram/stft/stft_tf.signal.stft/frame/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2H
Fsequential/melspectrogram/stft/stft_tf.signal.stft/frame/concat_1/axisј
Asequential/melspectrogram/stft/stft_tf.signal.stft/frame/concat_1ConcatV2Gsequential/melspectrogram/stft/stft_tf.signal.stft/frame/split:output:0Ssequential/melspectrogram/stft/stft_tf.signal.stft/frame/concat_1/values_1:output:0Gsequential/melspectrogram/stft/stft_tf.signal.stft/frame/split:output:2Osequential/melspectrogram/stft/stft_tf.signal.stft/frame/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2C
Asequential/melspectrogram/stft/stft_tf.signal.stft/frame/concat_1д
Csequential/melspectrogram/stft/stft_tf.signal.stft/frame/zeros_likeConst*
_output_shapes
:*
dtype0*
valueB: 2E
Csequential/melspectrogram/stft/stft_tf.signal.stft/frame/zeros_likeо
Hsequential/melspectrogram/stft/stft_tf.signal.stft/frame/ones_like/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2J
Hsequential/melspectrogram/stft/stft_tf.signal.stft/frame/ones_like/Shapeж
Hsequential/melspectrogram/stft/stft_tf.signal.stft/frame/ones_like/ConstConst*
_output_shapes
: *
dtype0*
value	B :2J
Hsequential/melspectrogram/stft/stft_tf.signal.stft/frame/ones_like/Constл
Bsequential/melspectrogram/stft/stft_tf.signal.stft/frame/ones_likeFillQsequential/melspectrogram/stft/stft_tf.signal.stft/frame/ones_like/Shape:output:0Qsequential/melspectrogram/stft/stft_tf.signal.stft/frame/ones_like/Const:output:0*
T0*
_output_shapes
:2D
Bsequential/melspectrogram/stft/stft_tf.signal.stft/frame/ones_like
Esequential/melspectrogram/stft/stft_tf.signal.stft/frame/StridedSliceStridedSlice,sequential/melspectrogram/stft/transpose:y:0Lsequential/melspectrogram/stft/stft_tf.signal.stft/frame/zeros_like:output:0Hsequential/melspectrogram/stft/stft_tf.signal.stft/frame/concat:output:0Ksequential/melspectrogram/stft/stft_tf.signal.stft/frame/ones_like:output:0*
Index0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ2G
Esequential/melspectrogram/stft/stft_tf.signal.stft/frame/StridedSliceќ
Bsequential/melspectrogram/stft/stft_tf.signal.stft/frame/Reshape_1ReshapeNsequential/melspectrogram/stft/stft_tf.signal.stft/frame/StridedSlice:output:0Jsequential/melspectrogram/stft/stft_tf.signal.stft/frame/concat_1:output:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2D
Bsequential/melspectrogram/stft/stft_tf.signal.stft/frame/Reshape_1в
Fsequential/melspectrogram/stft/stft_tf.signal.stft/frame/range_1/startConst*
_output_shapes
: *
dtype0*
value	B : 2H
Fsequential/melspectrogram/stft/stft_tf.signal.stft/frame/range_1/startв
Fsequential/melspectrogram/stft/stft_tf.signal.stft/frame/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :2H
Fsequential/melspectrogram/stft/stft_tf.signal.stft/frame/range_1/delta
@sequential/melspectrogram/stft/stft_tf.signal.stft/frame/range_1RangeOsequential/melspectrogram/stft/stft_tf.signal.stft/frame/range_1/start:output:0Dsequential/melspectrogram/stft/stft_tf.signal.stft/frame/Maximum:z:0Osequential/melspectrogram/stft/stft_tf.signal.stft/frame/range_1/delta:output:0*#
_output_shapes
:џџџџџџџџџ2B
@sequential/melspectrogram/stft/stft_tf.signal.stft/frame/range_1Щ
>sequential/melspectrogram/stft/stft_tf.signal.stft/frame/mul_1MulIsequential/melspectrogram/stft/stft_tf.signal.stft/frame/range_1:output:0Gsequential/melspectrogram/stft/stft_tf.signal.stft/frame/floordiv_2:z:0*
T0*#
_output_shapes
:џџџџџџџџџ2@
>sequential/melspectrogram/stft/stft_tf.signal.stft/frame/mul_1к
Jsequential/melspectrogram/stft/stft_tf.signal.stft/frame/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2L
Jsequential/melspectrogram/stft/stft_tf.signal.stft/frame/Reshape_2/shape/1х
Hsequential/melspectrogram/stft/stft_tf.signal.stft/frame/Reshape_2/shapePackDsequential/melspectrogram/stft/stft_tf.signal.stft/frame/Maximum:z:0Ssequential/melspectrogram/stft/stft_tf.signal.stft/frame/Reshape_2/shape/1:output:0*
N*
T0*
_output_shapes
:2J
Hsequential/melspectrogram/stft/stft_tf.signal.stft/frame/Reshape_2/shapeм
Bsequential/melspectrogram/stft/stft_tf.signal.stft/frame/Reshape_2ReshapeBsequential/melspectrogram/stft/stft_tf.signal.stft/frame/mul_1:z:0Qsequential/melspectrogram/stft/stft_tf.signal.stft/frame/Reshape_2/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2D
Bsequential/melspectrogram/stft/stft_tf.signal.stft/frame/Reshape_2в
Fsequential/melspectrogram/stft/stft_tf.signal.stft/frame/range_2/startConst*
_output_shapes
: *
dtype0*
value	B : 2H
Fsequential/melspectrogram/stft/stft_tf.signal.stft/frame/range_2/startв
Fsequential/melspectrogram/stft/stft_tf.signal.stft/frame/range_2/deltaConst*
_output_shapes
: *
dtype0*
value	B :2H
Fsequential/melspectrogram/stft/stft_tf.signal.stft/frame/range_2/delta
@sequential/melspectrogram/stft/stft_tf.signal.stft/frame/range_2RangeOsequential/melspectrogram/stft/stft_tf.signal.stft/frame/range_2/start:output:0Gsequential/melspectrogram/stft/stft_tf.signal.stft/frame/floordiv_1:z:0Osequential/melspectrogram/stft/stft_tf.signal.stft/frame/range_2/delta:output:0*
_output_shapes
:2B
@sequential/melspectrogram/stft/stft_tf.signal.stft/frame/range_2к
Jsequential/melspectrogram/stft/stft_tf.signal.stft/frame/Reshape_3/shape/0Const*
_output_shapes
: *
dtype0*
value	B :2L
Jsequential/melspectrogram/stft/stft_tf.signal.stft/frame/Reshape_3/shape/0ш
Hsequential/melspectrogram/stft/stft_tf.signal.stft/frame/Reshape_3/shapePackSsequential/melspectrogram/stft/stft_tf.signal.stft/frame/Reshape_3/shape/0:output:0Gsequential/melspectrogram/stft/stft_tf.signal.stft/frame/floordiv_1:z:0*
N*
T0*
_output_shapes
:2J
Hsequential/melspectrogram/stft/stft_tf.signal.stft/frame/Reshape_3/shapeк
Bsequential/melspectrogram/stft/stft_tf.signal.stft/frame/Reshape_3ReshapeIsequential/melspectrogram/stft/stft_tf.signal.stft/frame/range_2:output:0Qsequential/melspectrogram/stft/stft_tf.signal.stft/frame/Reshape_3/shape:output:0*
T0*
_output_shapes

:2D
Bsequential/melspectrogram/stft/stft_tf.signal.stft/frame/Reshape_3е
>sequential/melspectrogram/stft/stft_tf.signal.stft/frame/add_1AddV2Ksequential/melspectrogram/stft/stft_tf.signal.stft/frame/Reshape_2:output:0Ksequential/melspectrogram/stft/stft_tf.signal.stft/frame/Reshape_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2@
>sequential/melspectrogram/stft/stft_tf.signal.stft/frame/add_1ш
Asequential/melspectrogram/stft/stft_tf.signal.stft/frame/GatherV2GatherV2Ksequential/melspectrogram/stft/stft_tf.signal.stft/frame/Reshape_1:output:0Bsequential/melspectrogram/stft/stft_tf.signal.stft/frame/add_1:z:0Osequential/melspectrogram/stft/stft_tf.signal.stft/frame/strided_slice:output:0*
Taxis0*
Tindices0*
Tparams0*F
_output_shapes4
2:0џџџџџџџџџџџџџџџџџџџџџџџџџџџ2C
Asequential/melspectrogram/stft/stft_tf.signal.stft/frame/GatherV2о
Jsequential/melspectrogram/stft/stft_tf.signal.stft/frame/concat_2/values_1PackDsequential/melspectrogram/stft/stft_tf.signal.stft/frame/Maximum:z:0Hsequential/melspectrogram/stft/stft_tf.signal.stft/frame_length:output:0*
N*
T0*
_output_shapes
:2L
Jsequential/melspectrogram/stft/stft_tf.signal.stft/frame/concat_2/values_1в
Fsequential/melspectrogram/stft/stft_tf.signal.stft/frame/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2H
Fsequential/melspectrogram/stft/stft_tf.signal.stft/frame/concat_2/axisј
Asequential/melspectrogram/stft/stft_tf.signal.stft/frame/concat_2ConcatV2Gsequential/melspectrogram/stft/stft_tf.signal.stft/frame/split:output:0Ssequential/melspectrogram/stft/stft_tf.signal.stft/frame/concat_2/values_1:output:0Gsequential/melspectrogram/stft/stft_tf.signal.stft/frame/split:output:2Osequential/melspectrogram/stft/stft_tf.signal.stft/frame/concat_2/axis:output:0*
N*
T0*
_output_shapes
:2C
Asequential/melspectrogram/stft/stft_tf.signal.stft/frame/concat_2ч
Bsequential/melspectrogram/stft/stft_tf.signal.stft/frame/Reshape_4ReshapeJsequential/melspectrogram/stft/stft_tf.signal.stft/frame/GatherV2:output:0Jsequential/melspectrogram/stft/stft_tf.signal.stft/frame/concat_2:output:0*
T0*1
_output_shapes
:џџџџџџџџџЗ2D
Bsequential/melspectrogram/stft/stft_tf.signal.stft/frame/Reshape_4д
Gsequential/melspectrogram/stft/stft_tf.signal.stft/hann_window/periodicConst*
_output_shapes
: *
dtype0
*
value	B
 Z2I
Gsequential/melspectrogram/stft/stft_tf.signal.stft/hann_window/periodic
Csequential/melspectrogram/stft/stft_tf.signal.stft/hann_window/CastCastPsequential/melspectrogram/stft/stft_tf.signal.stft/hann_window/periodic:output:0*

DstT0*

SrcT0
*
_output_shapes
: 2E
Csequential/melspectrogram/stft/stft_tf.signal.stft/hann_window/Castи
Isequential/melspectrogram/stft/stft_tf.signal.stft/hann_window/FloorMod/yConst*
_output_shapes
: *
dtype0*
value	B :2K
Isequential/melspectrogram/stft/stft_tf.signal.stft/hann_window/FloorMod/yн
Gsequential/melspectrogram/stft/stft_tf.signal.stft/hann_window/FloorModFloorModHsequential/melspectrogram/stft/stft_tf.signal.stft/frame_length:output:0Rsequential/melspectrogram/stft/stft_tf.signal.stft/hann_window/FloorMod/y:output:0*
T0*
_output_shapes
: 2I
Gsequential/melspectrogram/stft/stft_tf.signal.stft/hann_window/FloorModЮ
Dsequential/melspectrogram/stft/stft_tf.signal.stft/hann_window/sub/xConst*
_output_shapes
: *
dtype0*
value	B :2F
Dsequential/melspectrogram/stft/stft_tf.signal.stft/hann_window/sub/xЬ
Bsequential/melspectrogram/stft/stft_tf.signal.stft/hann_window/subSubMsequential/melspectrogram/stft/stft_tf.signal.stft/hann_window/sub/x:output:0Ksequential/melspectrogram/stft/stft_tf.signal.stft/hann_window/FloorMod:z:0*
T0*
_output_shapes
: 2D
Bsequential/melspectrogram/stft/stft_tf.signal.stft/hann_window/subС
Bsequential/melspectrogram/stft/stft_tf.signal.stft/hann_window/mulMulGsequential/melspectrogram/stft/stft_tf.signal.stft/hann_window/Cast:y:0Fsequential/melspectrogram/stft/stft_tf.signal.stft/hann_window/sub:z:0*
T0*
_output_shapes
: 2D
Bsequential/melspectrogram/stft/stft_tf.signal.stft/hann_window/mulФ
Bsequential/melspectrogram/stft/stft_tf.signal.stft/hann_window/addAddV2Hsequential/melspectrogram/stft/stft_tf.signal.stft/frame_length:output:0Fsequential/melspectrogram/stft/stft_tf.signal.stft/hann_window/mul:z:0*
T0*
_output_shapes
: 2D
Bsequential/melspectrogram/stft/stft_tf.signal.stft/hann_window/addв
Fsequential/melspectrogram/stft/stft_tf.signal.stft/hann_window/sub_1/yConst*
_output_shapes
: *
dtype0*
value	B :2H
Fsequential/melspectrogram/stft/stft_tf.signal.stft/hann_window/sub_1/yЭ
Dsequential/melspectrogram/stft/stft_tf.signal.stft/hann_window/sub_1SubFsequential/melspectrogram/stft/stft_tf.signal.stft/hann_window/add:z:0Osequential/melspectrogram/stft/stft_tf.signal.stft/hann_window/sub_1/y:output:0*
T0*
_output_shapes
: 2F
Dsequential/melspectrogram/stft/stft_tf.signal.stft/hann_window/sub_1
Esequential/melspectrogram/stft/stft_tf.signal.stft/hann_window/Cast_1CastHsequential/melspectrogram/stft/stft_tf.signal.stft/hann_window/sub_1:z:0*

DstT0*

SrcT0*
_output_shapes
: 2G
Esequential/melspectrogram/stft/stft_tf.signal.stft/hann_window/Cast_1к
Jsequential/melspectrogram/stft/stft_tf.signal.stft/hann_window/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2L
Jsequential/melspectrogram/stft/stft_tf.signal.stft/hann_window/range/startк
Jsequential/melspectrogram/stft/stft_tf.signal.stft/hann_window/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2L
Jsequential/melspectrogram/stft/stft_tf.signal.stft/hann_window/range/deltaІ
Dsequential/melspectrogram/stft/stft_tf.signal.stft/hann_window/rangeRangeSsequential/melspectrogram/stft/stft_tf.signal.stft/hann_window/range/start:output:0Hsequential/melspectrogram/stft/stft_tf.signal.stft/frame_length:output:0Ssequential/melspectrogram/stft/stft_tf.signal.stft/hann_window/range/delta:output:0*
_output_shapes	
:2F
Dsequential/melspectrogram/stft/stft_tf.signal.stft/hann_window/range
Esequential/melspectrogram/stft/stft_tf.signal.stft/hann_window/Cast_2CastMsequential/melspectrogram/stft/stft_tf.signal.stft/hann_window/range:output:0*

DstT0*

SrcT0*
_output_shapes	
:2G
Esequential/melspectrogram/stft/stft_tf.signal.stft/hann_window/Cast_2б
Dsequential/melspectrogram/stft/stft_tf.signal.stft/hann_window/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *лЩ@2F
Dsequential/melspectrogram/stft/stft_tf.signal.stft/hann_window/Constг
Dsequential/melspectrogram/stft/stft_tf.signal.stft/hann_window/mul_1MulMsequential/melspectrogram/stft/stft_tf.signal.stft/hann_window/Const:output:0Isequential/melspectrogram/stft/stft_tf.signal.stft/hann_window/Cast_2:y:0*
T0*
_output_shapes	
:2F
Dsequential/melspectrogram/stft/stft_tf.signal.stft/hann_window/mul_1ж
Fsequential/melspectrogram/stft/stft_tf.signal.stft/hann_window/truedivRealDivHsequential/melspectrogram/stft/stft_tf.signal.stft/hann_window/mul_1:z:0Isequential/melspectrogram/stft/stft_tf.signal.stft/hann_window/Cast_1:y:0*
T0*
_output_shapes	
:2H
Fsequential/melspectrogram/stft/stft_tf.signal.stft/hann_window/truediv
Bsequential/melspectrogram/stft/stft_tf.signal.stft/hann_window/CosCosJsequential/melspectrogram/stft/stft_tf.signal.stft/hann_window/truediv:z:0*
T0*
_output_shapes	
:2D
Bsequential/melspectrogram/stft/stft_tf.signal.stft/hann_window/Cosе
Fsequential/melspectrogram/stft/stft_tf.signal.stft/hann_window/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2H
Fsequential/melspectrogram/stft/stft_tf.signal.stft/hann_window/mul_2/xв
Dsequential/melspectrogram/stft/stft_tf.signal.stft/hann_window/mul_2MulOsequential/melspectrogram/stft/stft_tf.signal.stft/hann_window/mul_2/x:output:0Fsequential/melspectrogram/stft/stft_tf.signal.stft/hann_window/Cos:y:0*
T0*
_output_shapes	
:2F
Dsequential/melspectrogram/stft/stft_tf.signal.stft/hann_window/mul_2е
Fsequential/melspectrogram/stft/stft_tf.signal.stft/hann_window/sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2H
Fsequential/melspectrogram/stft/stft_tf.signal.stft/hann_window/sub_2/xд
Dsequential/melspectrogram/stft/stft_tf.signal.stft/hann_window/sub_2SubOsequential/melspectrogram/stft/stft_tf.signal.stft/hann_window/sub_2/x:output:0Hsequential/melspectrogram/stft/stft_tf.signal.stft/hann_window/mul_2:z:0*
T0*
_output_shapes	
:2F
Dsequential/melspectrogram/stft/stft_tf.signal.stft/hann_window/sub_2Ъ
6sequential/melspectrogram/stft/stft_tf.signal.stft/mulMulKsequential/melspectrogram/stft/stft_tf.signal.stft/frame/Reshape_4:output:0Hsequential/melspectrogram/stft/stft_tf.signal.stft/hann_window/sub_2:z:0*
T0*1
_output_shapes
:џџџџџџџџџЗ28
6sequential/melspectrogram/stft/stft_tf.signal.stft/mulў
>sequential/melspectrogram/stft/stft_tf.signal.stft/rfft/packedPackFsequential/melspectrogram/stft/stft_tf.signal.stft/fft_length:output:0*
N*
T0*
_output_shapes
:2@
>sequential/melspectrogram/stft/stft_tf.signal.stft/rfft/packedг
Bsequential/melspectrogram/stft/stft_tf.signal.stft/rfft/fft_lengthConst*
_output_shapes
:*
dtype0*
valueB:2D
Bsequential/melspectrogram/stft/stft_tf.signal.stft/rfft/fft_lengthЖ
7sequential/melspectrogram/stft/stft_tf.signal.stft/rfftRFFT:sequential/melspectrogram/stft/stft_tf.signal.stft/mul:z:0Ksequential/melspectrogram/stft/stft_tf.signal.stft/rfft/fft_length:output:0*1
_output_shapes
:џџџџџџџџџЗ29
7sequential/melspectrogram/stft/stft_tf.signal.stft/rfftЛ
/sequential/melspectrogram/stft/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             21
/sequential/melspectrogram/stft/transpose_1/perm
*sequential/melspectrogram/stft/transpose_1	Transpose@sequential/melspectrogram/stft/stft_tf.signal.stft/rfft:output:08sequential/melspectrogram/stft/transpose_1/perm:output:0*
T0*1
_output_shapes
:џџџџџџџџџЗ2,
*sequential/melspectrogram/stft/transpose_1У
'sequential/melspectrogram/magnitude/Abs
ComplexAbs.sequential/melspectrogram/stft/transpose_1:y:0*1
_output_shapes
:џџџџџџџџџЗ2)
'sequential/melspectrogram/magnitude/AbsР
9sequential/melspectrogram/apply_filterbank/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2;
9sequential/melspectrogram/apply_filterbank/Tensordot/axesЫ
9sequential/melspectrogram/apply_filterbank/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          2;
9sequential/melspectrogram/apply_filterbank/Tensordot/freeг
:sequential/melspectrogram/apply_filterbank/Tensordot/ShapeShape+sequential/melspectrogram/magnitude/Abs:y:0*
T0*
_output_shapes
:2<
:sequential/melspectrogram/apply_filterbank/Tensordot/ShapeЪ
Bsequential/melspectrogram/apply_filterbank/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2D
Bsequential/melspectrogram/apply_filterbank/Tensordot/GatherV2/axisЈ
=sequential/melspectrogram/apply_filterbank/Tensordot/GatherV2GatherV2Csequential/melspectrogram/apply_filterbank/Tensordot/Shape:output:0Bsequential/melspectrogram/apply_filterbank/Tensordot/free:output:0Ksequential/melspectrogram/apply_filterbank/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2?
=sequential/melspectrogram/apply_filterbank/Tensordot/GatherV2Ю
Dsequential/melspectrogram/apply_filterbank/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2F
Dsequential/melspectrogram/apply_filterbank/Tensordot/GatherV2_1/axisЎ
?sequential/melspectrogram/apply_filterbank/Tensordot/GatherV2_1GatherV2Csequential/melspectrogram/apply_filterbank/Tensordot/Shape:output:0Bsequential/melspectrogram/apply_filterbank/Tensordot/axes:output:0Msequential/melspectrogram/apply_filterbank/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2A
?sequential/melspectrogram/apply_filterbank/Tensordot/GatherV2_1Т
:sequential/melspectrogram/apply_filterbank/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2<
:sequential/melspectrogram/apply_filterbank/Tensordot/ConstЌ
9sequential/melspectrogram/apply_filterbank/Tensordot/ProdProdFsequential/melspectrogram/apply_filterbank/Tensordot/GatherV2:output:0Csequential/melspectrogram/apply_filterbank/Tensordot/Const:output:0*
T0*
_output_shapes
: 2;
9sequential/melspectrogram/apply_filterbank/Tensordot/ProdЦ
<sequential/melspectrogram/apply_filterbank/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2>
<sequential/melspectrogram/apply_filterbank/Tensordot/Const_1Д
;sequential/melspectrogram/apply_filterbank/Tensordot/Prod_1ProdHsequential/melspectrogram/apply_filterbank/Tensordot/GatherV2_1:output:0Esequential/melspectrogram/apply_filterbank/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2=
;sequential/melspectrogram/apply_filterbank/Tensordot/Prod_1Ц
@sequential/melspectrogram/apply_filterbank/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2B
@sequential/melspectrogram/apply_filterbank/Tensordot/concat/axis
;sequential/melspectrogram/apply_filterbank/Tensordot/concatConcatV2Bsequential/melspectrogram/apply_filterbank/Tensordot/free:output:0Bsequential/melspectrogram/apply_filterbank/Tensordot/axes:output:0Isequential/melspectrogram/apply_filterbank/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2=
;sequential/melspectrogram/apply_filterbank/Tensordot/concatИ
:sequential/melspectrogram/apply_filterbank/Tensordot/stackPackBsequential/melspectrogram/apply_filterbank/Tensordot/Prod:output:0Dsequential/melspectrogram/apply_filterbank/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2<
:sequential/melspectrogram/apply_filterbank/Tensordot/stackМ
>sequential/melspectrogram/apply_filterbank/Tensordot/transpose	Transpose+sequential/melspectrogram/magnitude/Abs:y:0Dsequential/melspectrogram/apply_filterbank/Tensordot/concat:output:0*
T0*1
_output_shapes
:џџџџџџџџџЗ2@
>sequential/melspectrogram/apply_filterbank/Tensordot/transposeЫ
<sequential/melspectrogram/apply_filterbank/Tensordot/ReshapeReshapeBsequential/melspectrogram/apply_filterbank/Tensordot/transpose:y:0Csequential/melspectrogram/apply_filterbank/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2>
<sequential/melspectrogram/apply_filterbank/Tensordot/ReshapeЖ
;sequential/melspectrogram/apply_filterbank/Tensordot/MatMulMatMulEsequential/melspectrogram/apply_filterbank/Tensordot/Reshape:output:06sequential_melspectrogram_apply_filterbank_tensordot_b*
T0*(
_output_shapes
:џџџџџџџџџ2=
;sequential/melspectrogram/apply_filterbank/Tensordot/MatMulЧ
<sequential/melspectrogram/apply_filterbank/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2>
<sequential/melspectrogram/apply_filterbank/Tensordot/Const_2Ъ
Bsequential/melspectrogram/apply_filterbank/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2D
Bsequential/melspectrogram/apply_filterbank/Tensordot/concat_1/axis
=sequential/melspectrogram/apply_filterbank/Tensordot/concat_1ConcatV2Fsequential/melspectrogram/apply_filterbank/Tensordot/GatherV2:output:0Esequential/melspectrogram/apply_filterbank/Tensordot/Const_2:output:0Ksequential/melspectrogram/apply_filterbank/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2?
=sequential/melspectrogram/apply_filterbank/Tensordot/concat_1Т
4sequential/melspectrogram/apply_filterbank/TensordotReshapeEsequential/melspectrogram/apply_filterbank/Tensordot/MatMul:product:0Fsequential/melspectrogram/apply_filterbank/Tensordot/concat_1:output:0*
T0*1
_output_shapes
:џџџџџџџџџЗ26
4sequential/melspectrogram/apply_filterbank/TensordotЯ
9sequential/melspectrogram/apply_filterbank/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             2;
9sequential/melspectrogram/apply_filterbank/transpose/permИ
4sequential/melspectrogram/apply_filterbank/transpose	Transpose=sequential/melspectrogram/apply_filterbank/Tensordot:output:0Bsequential/melspectrogram/apply_filterbank/transpose/perm:output:0*
T0*1
_output_shapes
:џџџџџџџџџЗ26
4sequential/melspectrogram/apply_filterbank/transposeб
-sequential/batch_normalization/ReadVariableOpReadVariableOp6sequential_batch_normalization_readvariableop_resource*
_output_shapes
:*
dtype02/
-sequential/batch_normalization/ReadVariableOpз
/sequential/batch_normalization/ReadVariableOp_1ReadVariableOp8sequential_batch_normalization_readvariableop_1_resource*
_output_shapes
:*
dtype021
/sequential/batch_normalization/ReadVariableOp_1
>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpGsequential_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02@
>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp
@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpIsequential_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02B
@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1И
/sequential/batch_normalization/FusedBatchNormV3FusedBatchNormV38sequential/melspectrogram/apply_filterbank/transpose:y:05sequential/batch_normalization/ReadVariableOp:value:07sequential/batch_normalization/ReadVariableOp_1:value:0Fsequential/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0Hsequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:џџџџџџџџџЗ:::::*
epsilon%o:*
is_training( 21
/sequential/batch_normalization/FusedBatchNormV3Ы
'sequential/conv2d/Conv2D/ReadVariableOpReadVariableOp0sequential_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02)
'sequential/conv2d/Conv2D/ReadVariableOp
sequential/conv2d/Conv2DConv2D3sequential/batch_normalization/FusedBatchNormV3:y:0/sequential/conv2d/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџЗ*
paddingSAME*
strides
2
sequential/conv2d/Conv2DТ
(sequential/conv2d/BiasAdd/ReadVariableOpReadVariableOp1sequential_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(sequential/conv2d/BiasAdd/ReadVariableOpв
sequential/conv2d/BiasAddBiasAdd!sequential/conv2d/Conv2D:output:00sequential/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџЗ2
sequential/conv2d/BiasAdd
sequential/conv2d/ReluRelu"sequential/conv2d/BiasAdd:output:0*
T0*1
_output_shapes
:џџџџџџџџџЗ2
sequential/conv2d/Reluт
 sequential/max_pooling2d/MaxPoolMaxPool$sequential/conv2d/Relu:activations:0*/
_output_shapes
:џџџџџџџџџg@*
ksize
*
paddingVALID*
strides
2"
 sequential/max_pooling2d/MaxPoolз
/sequential/batch_normalization_1/ReadVariableOpReadVariableOp8sequential_batch_normalization_1_readvariableop_resource*
_output_shapes
:*
dtype021
/sequential/batch_normalization_1/ReadVariableOpн
1sequential/batch_normalization_1/ReadVariableOp_1ReadVariableOp:sequential_batch_normalization_1_readvariableop_1_resource*
_output_shapes
:*
dtype023
1sequential/batch_normalization_1/ReadVariableOp_1
@sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOpIsequential_batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02B
@sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp
Bsequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpKsequential_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02D
Bsequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1Г
1sequential/batch_normalization_1/FusedBatchNormV3FusedBatchNormV3)sequential/max_pooling2d/MaxPool:output:07sequential/batch_normalization_1/ReadVariableOp:value:09sequential/batch_normalization_1/ReadVariableOp_1:value:0Hsequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0Jsequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџg@:::::*
epsilon%o:*
is_training( 23
1sequential/batch_normalization_1/FusedBatchNormV3б
)sequential/conv2d_1/Conv2D/ReadVariableOpReadVariableOp2sequential_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02+
)sequential/conv2d_1/Conv2D/ReadVariableOp
sequential/conv2d_1/Conv2DConv2D5sequential/batch_normalization_1/FusedBatchNormV3:y:01sequential/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџg@*
paddingSAME*
strides
2
sequential/conv2d_1/Conv2DШ
*sequential/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp3sequential_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02,
*sequential/conv2d_1/BiasAdd/ReadVariableOpи
sequential/conv2d_1/BiasAddBiasAdd#sequential/conv2d_1/Conv2D:output:02sequential/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџg@2
sequential/conv2d_1/BiasAdd
sequential/conv2d_1/ReluRelu$sequential/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџg@2
sequential/conv2d_1/Reluш
"sequential/max_pooling2d_1/MaxPoolMaxPool&sequential/conv2d_1/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ" *
ksize
*
paddingVALID*
strides
2$
"sequential/max_pooling2d_1/MaxPoolз
/sequential/batch_normalization_2/ReadVariableOpReadVariableOp8sequential_batch_normalization_2_readvariableop_resource*
_output_shapes
:*
dtype021
/sequential/batch_normalization_2/ReadVariableOpн
1sequential/batch_normalization_2/ReadVariableOp_1ReadVariableOp:sequential_batch_normalization_2_readvariableop_1_resource*
_output_shapes
:*
dtype023
1sequential/batch_normalization_2/ReadVariableOp_1
@sequential/batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOpIsequential_batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02B
@sequential/batch_normalization_2/FusedBatchNormV3/ReadVariableOp
Bsequential/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpKsequential_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02D
Bsequential/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1Е
1sequential/batch_normalization_2/FusedBatchNormV3FusedBatchNormV3+sequential/max_pooling2d_1/MaxPool:output:07sequential/batch_normalization_2/ReadVariableOp:value:09sequential/batch_normalization_2/ReadVariableOp_1:value:0Hsequential/batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0Jsequential/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ" :::::*
epsilon%o:*
is_training( 23
1sequential/batch_normalization_2/FusedBatchNormV3б
)sequential/conv2d_2/Conv2D/ReadVariableOpReadVariableOp2sequential_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02+
)sequential/conv2d_2/Conv2D/ReadVariableOp
sequential/conv2d_2/Conv2DConv2D5sequential/batch_normalization_2/FusedBatchNormV3:y:01sequential/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ"  *
paddingSAME*
strides
2
sequential/conv2d_2/Conv2DШ
*sequential/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp3sequential_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02,
*sequential/conv2d_2/BiasAdd/ReadVariableOpи
sequential/conv2d_2/BiasAddBiasAdd#sequential/conv2d_2/Conv2D:output:02sequential/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ"  2
sequential/conv2d_2/BiasAdd
sequential/conv2d_2/ReluRelu$sequential/conv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ"  2
sequential/conv2d_2/Reluш
"sequential/max_pooling2d_2/MaxPoolMaxPool&sequential/conv2d_2/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ *
ksize
*
paddingVALID*
strides
2$
"sequential/max_pooling2d_2/MaxPoolз
/sequential/batch_normalization_3/ReadVariableOpReadVariableOp8sequential_batch_normalization_3_readvariableop_resource*
_output_shapes
: *
dtype021
/sequential/batch_normalization_3/ReadVariableOpн
1sequential/batch_normalization_3/ReadVariableOp_1ReadVariableOp:sequential_batch_normalization_3_readvariableop_1_resource*
_output_shapes
: *
dtype023
1sequential/batch_normalization_3/ReadVariableOp_1
@sequential/batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOpIsequential_batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02B
@sequential/batch_normalization_3/FusedBatchNormV3/ReadVariableOp
Bsequential/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpKsequential_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02D
Bsequential/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1Е
1sequential/batch_normalization_3/FusedBatchNormV3FusedBatchNormV3+sequential/max_pooling2d_2/MaxPool:output:07sequential/batch_normalization_3/ReadVariableOp:value:09sequential/batch_normalization_3/ReadVariableOp_1:value:0Hsequential/batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0Jsequential/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ : : : : :*
epsilon%o:*
is_training( 23
1sequential/batch_normalization_3/FusedBatchNormV3
sequential/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ "  2
sequential/flatten/Constа
sequential/flatten/ReshapeReshape5sequential/batch_normalization_3/FusedBatchNormV3:y:0!sequential/flatten/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџD2
sequential/flatten/ReshapeС
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource*
_output_shapes
:	D@*
dtype02(
&sequential/dense/MatMul/ReadVariableOpУ
sequential/dense/MatMulMatMul#sequential/flatten/Reshape:output:0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
sequential/dense/MatMulП
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02)
'sequential/dense/BiasAdd/ReadVariableOpХ
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
sequential/dense/BiasAdd
sequential/dense/ReluRelu!sequential/dense/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
sequential/dense/Relu
sequential/dropout/IdentityIdentity#sequential/dense/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
sequential/dropout/IdentityЦ
(sequential/dense_1/MatMul/ReadVariableOpReadVariableOp1sequential_dense_1_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02*
(sequential/dense_1/MatMul/ReadVariableOpЪ
sequential/dense_1/MatMulMatMul$sequential/dropout/Identity:output:00sequential/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
sequential/dense_1/MatMulХ
)sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)sequential/dense_1/BiasAdd/ReadVariableOpЭ
sequential/dense_1/BiasAddBiasAdd#sequential/dense_1/MatMul:product:01sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
sequential/dense_1/BiasAdd
sequential/dense_1/ReluRelu#sequential/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
sequential/dense_1/ReluЃ
sequential/dropout_1/IdentityIdentity%sequential/dense_1/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ2
sequential/dropout_1/IdentityЦ
(sequential/dense_2/MatMul/ReadVariableOpReadVariableOp1sequential_dense_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype02*
(sequential/dense_2/MatMul/ReadVariableOpЬ
sequential/dense_2/MatMulMatMul&sequential/dropout_1/Identity:output:00sequential/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
sequential/dense_2/MatMulХ
)sequential/dense_2/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)sequential/dense_2/BiasAdd/ReadVariableOpЭ
sequential/dense_2/BiasAddBiasAdd#sequential/dense_2/MatMul:product:01sequential/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
sequential/dense_2/BiasAdd
sequential/dense_2/SigmoidSigmoid#sequential/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
sequential/dense_2/SigmoidЊ
IdentityIdentitysequential/dense_2/Sigmoid:y:0?^sequential/batch_normalization/FusedBatchNormV3/ReadVariableOpA^sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1.^sequential/batch_normalization/ReadVariableOp0^sequential/batch_normalization/ReadVariableOp_1A^sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOpC^sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_10^sequential/batch_normalization_1/ReadVariableOp2^sequential/batch_normalization_1/ReadVariableOp_1A^sequential/batch_normalization_2/FusedBatchNormV3/ReadVariableOpC^sequential/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_10^sequential/batch_normalization_2/ReadVariableOp2^sequential/batch_normalization_2/ReadVariableOp_1A^sequential/batch_normalization_3/FusedBatchNormV3/ReadVariableOpC^sequential/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_10^sequential/batch_normalization_3/ReadVariableOp2^sequential/batch_normalization_3/ReadVariableOp_1)^sequential/conv2d/BiasAdd/ReadVariableOp(^sequential/conv2d/Conv2D/ReadVariableOp+^sequential/conv2d_1/BiasAdd/ReadVariableOp*^sequential/conv2d_1/Conv2D/ReadVariableOp+^sequential/conv2d_2/BiasAdd/ReadVariableOp*^sequential/conv2d_2/Conv2D/ReadVariableOp(^sequential/dense/BiasAdd/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp*^sequential/dense_1/BiasAdd/ReadVariableOp)^sequential/dense_1/MatMul/ReadVariableOp*^sequential/dense_2/BiasAdd/ReadVariableOp)^sequential/dense_2/MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*І
_input_shapes
:џџџџџџџџџт	:
::::::::::::::::::::::::::::2
>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp2
@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_12^
-sequential/batch_normalization/ReadVariableOp-sequential/batch_normalization/ReadVariableOp2b
/sequential/batch_normalization/ReadVariableOp_1/sequential/batch_normalization/ReadVariableOp_12
@sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp@sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp2
Bsequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1Bsequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12b
/sequential/batch_normalization_1/ReadVariableOp/sequential/batch_normalization_1/ReadVariableOp2f
1sequential/batch_normalization_1/ReadVariableOp_11sequential/batch_normalization_1/ReadVariableOp_12
@sequential/batch_normalization_2/FusedBatchNormV3/ReadVariableOp@sequential/batch_normalization_2/FusedBatchNormV3/ReadVariableOp2
Bsequential/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1Bsequential/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12b
/sequential/batch_normalization_2/ReadVariableOp/sequential/batch_normalization_2/ReadVariableOp2f
1sequential/batch_normalization_2/ReadVariableOp_11sequential/batch_normalization_2/ReadVariableOp_12
@sequential/batch_normalization_3/FusedBatchNormV3/ReadVariableOp@sequential/batch_normalization_3/FusedBatchNormV3/ReadVariableOp2
Bsequential/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1Bsequential/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12b
/sequential/batch_normalization_3/ReadVariableOp/sequential/batch_normalization_3/ReadVariableOp2f
1sequential/batch_normalization_3/ReadVariableOp_11sequential/batch_normalization_3/ReadVariableOp_12T
(sequential/conv2d/BiasAdd/ReadVariableOp(sequential/conv2d/BiasAdd/ReadVariableOp2R
'sequential/conv2d/Conv2D/ReadVariableOp'sequential/conv2d/Conv2D/ReadVariableOp2X
*sequential/conv2d_1/BiasAdd/ReadVariableOp*sequential/conv2d_1/BiasAdd/ReadVariableOp2V
)sequential/conv2d_1/Conv2D/ReadVariableOp)sequential/conv2d_1/Conv2D/ReadVariableOp2X
*sequential/conv2d_2/BiasAdd/ReadVariableOp*sequential/conv2d_2/BiasAdd/ReadVariableOp2V
)sequential/conv2d_2/Conv2D/ReadVariableOp)sequential/conv2d_2/Conv2D/ReadVariableOp2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2P
&sequential/dense/MatMul/ReadVariableOp&sequential/dense/MatMul/ReadVariableOp2V
)sequential/dense_1/BiasAdd/ReadVariableOp)sequential/dense_1/BiasAdd/ReadVariableOp2T
(sequential/dense_1/MatMul/ReadVariableOp(sequential/dense_1/MatMul/ReadVariableOp2V
)sequential/dense_2/BiasAdd/ReadVariableOp)sequential/dense_2/BiasAdd/ReadVariableOp2T
(sequential/dense_2/MatMul/ReadVariableOp(sequential/dense_2/MatMul/ReadVariableOp:X T
)
_output_shapes
:џџџџџџџџџт	
'
_user_specified_namereshape_input:&"
 
_output_shapes
:


f
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_20207

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ћ
@
)__inference_magnitude_layer_call_fn_23212
x
identityЧ
PartitionedCallPartitionedCallx*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:џџџџџџџџџЗ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_magnitude_layer_call_and_return_conditional_losses_198902
PartitionedCallv
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:џџџџџџџџџЗ2

Identity"
identityIdentity:output:0*0
_input_shapes
:џџџџџџџџџЗ:T P
1
_output_shapes
:џџџџџџџџџЗ

_user_specified_namex

c
D__inference_dropout_1_layer_call_and_return_conditional_losses_20961

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeД
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>2
dropout/GreaterEqual/yО
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*&
_input_shapes
:џџџџџџџџџ:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ч
b
D__inference_dropout_1_layer_call_and_return_conditional_losses_23058

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:џџџџџџџџџ2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:џџџџџџџџџ:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs"БL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*И
serving_defaultЄ
I
reshape_input8
serving_default_reshape_input:0џџџџџџџџџт	;
dense_20
StatefulPartitionedCall:0џџџџџџџџџtensorflow/serving/predict:Є
х
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
layer-7
	layer_with_weights-4
	layer-8

layer_with_weights-5

layer-9
layer-10
layer_with_weights-6
layer-11
layer-12
layer_with_weights-7
layer-13
layer-14
layer_with_weights-8
layer-15
layer-16
layer_with_weights-9
layer-17
	optimizer
regularization_losses
	variables
trainable_variables
	keras_api

signatures
З_default_save_signature
+И&call_and_return_all_conditional_losses
Й__call__" 
_tf_keras_sequential{"class_name": "Sequential", "name": "sequential", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 160000]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "reshape_input"}}, {"class_name": "Reshape", "config": {"name": "reshape", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 160000]}, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [160000, 1]}}}, {"class_name": "Sequential", "config": {"name": "melspectrogram", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 160000, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "stft_input"}}, {"class_name": "STFT", "config": {"name": "stft", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 160000, 1]}, "dtype": "float32", "n_fft": 1024, "win_length": 1024, "hop_length": 512, "window_name": null, "pad_begin": false, "pad_end": false, "input_data_format": "channels_last", "output_data_format": "channels_last"}}, {"class_name": "Magnitude", "config": {"name": "magnitude", "trainable": true, "dtype": "float32"}}, {"class_name": "ApplyFilterbank", "config": {"name": "apply_filterbank", "trainable": true, "dtype": "float32", "type": "mel", "filterbank_kwargs": {"sample_rate": 16000, "n_freq": 513, "n_mels": 128, "f_min": 0.0, "f_max": null, "htk": false, "norm": "slaney"}, "data_format": "channels_last"}}]}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [3, 2]}, "data_format": "channels_last"}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [3, 2]}, "data_format": "channels_last"}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_3", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 8, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 160000]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 160000]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "reshape_input"}}, {"class_name": "Reshape", "config": {"name": "reshape", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 160000]}, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [160000, 1]}}}, {"class_name": "Sequential", "config": {"name": "melspectrogram", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 160000, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "stft_input"}}, {"class_name": "STFT", "config": {"name": "stft", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 160000, 1]}, "dtype": "float32", "n_fft": 1024, "win_length": 1024, "hop_length": 512, "window_name": null, "pad_begin": false, "pad_end": false, "input_data_format": "channels_last", "output_data_format": "channels_last"}}, {"class_name": "Magnitude", "config": {"name": "magnitude", "trainable": true, "dtype": "float32"}}, {"class_name": "ApplyFilterbank", "config": {"name": "apply_filterbank", "trainable": true, "dtype": "float32", "type": "mel", "filterbank_kwargs": {"sample_rate": 16000, "n_freq": 513, "n_mels": 128, "f_min": 0.0, "f_max": null, "htk": false, "norm": "slaney"}, "data_format": "channels_last"}}]}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [3, 2]}, "data_format": "channels_last"}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [3, 2]}, "data_format": "channels_last"}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_3", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 8, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "binary_crossentropy", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "accuracy", "dtype": "float32", "fn": "binary_accuracy"}}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
ђ
regularization_losses
	variables
trainable_variables
	keras_api
+К&call_and_return_all_conditional_losses
Л__call__"с
_tf_keras_layerЧ{"class_name": "Reshape", "name": "reshape", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 160000]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "reshape", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 160000]}, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [160000, 1]}}}
Б
layer-0
layer-1
layer-2
 regularization_losses
!	variables
"trainable_variables
#	keras_api
+М&call_and_return_all_conditional_losses
Н__call__"љ
_tf_keras_sequentialк{"class_name": "Sequential", "name": "melspectrogram", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "melspectrogram", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 160000, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "stft_input"}}, {"class_name": "STFT", "config": {"name": "stft", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 160000, 1]}, "dtype": "float32", "n_fft": 1024, "win_length": 1024, "hop_length": 512, "window_name": null, "pad_begin": false, "pad_end": false, "input_data_format": "channels_last", "output_data_format": "channels_last"}}, {"class_name": "Magnitude", "config": {"name": "magnitude", "trainable": true, "dtype": "float32"}}, {"class_name": "ApplyFilterbank", "config": {"name": "apply_filterbank", "trainable": true, "dtype": "float32", "type": "mel", "filterbank_kwargs": {"sample_rate": 16000, "n_freq": 513, "n_mels": 128, "f_min": 0.0, "f_max": null, "htk": false, "norm": "slaney"}, "data_format": "channels_last"}}]}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 160000, 1]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "melspectrogram", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 160000, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "stft_input"}}, {"class_name": "STFT", "config": {"name": "stft", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 160000, 1]}, "dtype": "float32", "n_fft": 1024, "win_length": 1024, "hop_length": 512, "window_name": null, "pad_begin": false, "pad_end": false, "input_data_format": "channels_last", "output_data_format": "channels_last"}}, {"class_name": "Magnitude", "config": {"name": "magnitude", "trainable": true, "dtype": "float32"}}, {"class_name": "ApplyFilterbank", "config": {"name": "apply_filterbank", "trainable": true, "dtype": "float32", "type": "mel", "filterbank_kwargs": {"sample_rate": 16000, "n_freq": 513, "n_mels": 128, "f_min": 0.0, "f_max": null, "htk": false, "norm": "slaney"}, "data_format": "channels_last"}}]}}}
И	
$axis
	%gamma
&beta
'moving_mean
(moving_variance
)regularization_losses
*	variables
+trainable_variables
,	keras_api
+О&call_and_return_all_conditional_losses
П__call__"т
_tf_keras_layerШ{"class_name": "BatchNormalization", "name": "batch_normalization", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 311, 128, 1]}}
я	

-kernel
.bias
/regularization_losses
0	variables
1trainable_variables
2	keras_api
+Р&call_and_return_all_conditional_losses
С__call__"Ш
_tf_keras_layerЎ{"class_name": "Conv2D", "name": "conv2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 311, 128, 1]}}
§
3regularization_losses
4	variables
5trainable_variables
6	keras_api
+Т&call_and_return_all_conditional_losses
У__call__"ь
_tf_keras_layerв{"class_name": "MaxPooling2D", "name": "max_pooling2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [3, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
Л	
7axis
	8gamma
9beta
:moving_mean
;moving_variance
<regularization_losses
=	variables
>trainable_variables
?	keras_api
+Ф&call_and_return_all_conditional_losses
Х__call__"х
_tf_keras_layerЫ{"class_name": "BatchNormalization", "name": "batch_normalization_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 8}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 103, 64, 8]}}
ѓ	

@kernel
Abias
Bregularization_losses
C	variables
Dtrainable_variables
E	keras_api
+Ц&call_and_return_all_conditional_losses
Ч__call__"Ь
_tf_keras_layerВ{"class_name": "Conv2D", "name": "conv2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 8}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 103, 64, 8]}}

Fregularization_losses
G	variables
Htrainable_variables
I	keras_api
+Ш&call_and_return_all_conditional_losses
Щ__call__"№
_tf_keras_layerж{"class_name": "MaxPooling2D", "name": "max_pooling2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [3, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
М	
Jaxis
	Kgamma
Lbeta
Mmoving_mean
Nmoving_variance
Oregularization_losses
P	variables
Qtrainable_variables
R	keras_api
+Ъ&call_and_return_all_conditional_losses
Ы__call__"ц
_tf_keras_layerЬ{"class_name": "BatchNormalization", "name": "batch_normalization_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 34, 32, 16]}}
є	

Skernel
Tbias
Uregularization_losses
V	variables
Wtrainable_variables
X	keras_api
+Ь&call_and_return_all_conditional_losses
Э__call__"Э
_tf_keras_layerГ{"class_name": "Conv2D", "name": "conv2d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 34, 32, 16]}}

Yregularization_losses
Z	variables
[trainable_variables
\	keras_api
+Ю&call_and_return_all_conditional_losses
Я__call__"№
_tf_keras_layerж{"class_name": "MaxPooling2D", "name": "max_pooling2d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
М	
]axis
	^gamma
_beta
`moving_mean
amoving_variance
bregularization_losses
c	variables
dtrainable_variables
e	keras_api
+а&call_and_return_all_conditional_losses
б__call__"ц
_tf_keras_layerЬ{"class_name": "BatchNormalization", "name": "batch_normalization_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_3", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 17, 16, 32]}}
ф
fregularization_losses
g	variables
htrainable_variables
i	keras_api
+в&call_and_return_all_conditional_losses
г__call__"г
_tf_keras_layerЙ{"class_name": "Flatten", "name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
ђ

jkernel
kbias
lregularization_losses
m	variables
ntrainable_variables
o	keras_api
+д&call_and_return_all_conditional_losses
е__call__"Ы
_tf_keras_layerБ{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 8704}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 8704]}}
у
pregularization_losses
q	variables
rtrainable_variables
s	keras_api
+ж&call_and_return_all_conditional_losses
з__call__"в
_tf_keras_layerИ{"class_name": "Dropout", "name": "dropout", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}
ё

tkernel
ubias
vregularization_losses
w	variables
xtrainable_variables
y	keras_api
+и&call_and_return_all_conditional_losses
й__call__"Ъ
_tf_keras_layerА{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 8, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
ч
zregularization_losses
{	variables
|trainable_variables
}	keras_api
+к&call_and_return_all_conditional_losses
л__call__"ж
_tf_keras_layerМ{"class_name": "Dropout", "name": "dropout_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}
і

~kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
+м&call_and_return_all_conditional_losses
н__call__"Ы
_tf_keras_layerБ{"class_name": "Dense", "name": "dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 8}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 8]}}
ш
	iter
beta_1
beta_2

decay
learning_rate%m&m-m.m8m9m@mAmKmLmSmTm^m_mjmkmtmum ~mЁmЂ%vЃ&vЄ-vЅ.vІ8vЇ9vЈ@vЉAvЊKvЋLvЌSv­TvЎ^vЏ_vАjvБkvВtvГuvД~vЕvЖ"
	optimizer
 "
trackable_list_wrapper
і
%0
&1
'2
(3
-4
.5
86
97
:8
;9
@10
A11
K12
L13
M14
N15
S16
T17
^18
_19
`20
a21
j22
k23
t24
u25
~26
27"
trackable_list_wrapper
Ж
%0
&1
-2
.3
84
95
@6
A7
K8
L9
S10
T11
^12
_13
j14
k15
t16
u17
~18
19"
trackable_list_wrapper
г
regularization_losses
 layer_regularization_losses
layers
metrics
non_trainable_variables
	variables
layer_metrics
trainable_variables
Й__call__
З_default_save_signature
+И&call_and_return_all_conditional_losses
'И"call_and_return_conditional_losses"
_generic_user_object
-
оserving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
regularization_losses
 layer_regularization_losses
layers
metrics
non_trainable_variables
	variables
layer_metrics
trainable_variables
Л__call__
+К&call_and_return_all_conditional_losses
'К"call_and_return_conditional_losses"
_generic_user_object
ю
regularization_losses
	variables
trainable_variables
	keras_api
+п&call_and_return_all_conditional_losses
р__call__"й
_tf_keras_layerП{"class_name": "STFT", "name": "stft", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 160000, 1]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "stft", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 160000, 1]}, "dtype": "float32", "n_fft": 1024, "win_length": 1024, "hop_length": 512, "window_name": null, "pad_begin": false, "pad_end": false, "input_data_format": "channels_last", "output_data_format": "channels_last"}}
О
regularization_losses
	variables
trainable_variables
	keras_api
+с&call_and_return_all_conditional_losses
т__call__"Љ
_tf_keras_layer{"class_name": "Magnitude", "name": "magnitude", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "magnitude", "trainable": true, "dtype": "float32"}}
Ё
filterbank_kwargs
regularization_losses
	variables
trainable_variables
	keras_api
+у&call_and_return_all_conditional_losses
ф__call__"є
_tf_keras_layerк{"class_name": "ApplyFilterbank", "name": "apply_filterbank", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "apply_filterbank", "trainable": true, "dtype": "float32", "type": "mel", "filterbank_kwargs": {"sample_rate": 16000, "n_freq": 513, "n_mels": 128, "f_min": 0.0, "f_max": null, "htk": false, "norm": "slaney"}, "data_format": "channels_last"}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
 regularization_losses
  layer_regularization_losses
Ёlayers
Ђmetrics
Ѓnon_trainable_variables
!	variables
Єlayer_metrics
"trainable_variables
Н__call__
+М&call_and_return_all_conditional_losses
'М"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
':%2batch_normalization/gamma
&:$2batch_normalization/beta
/:- (2batch_normalization/moving_mean
3:1 (2#batch_normalization/moving_variance
 "
trackable_list_wrapper
<
%0
&1
'2
(3"
trackable_list_wrapper
.
%0
&1"
trackable_list_wrapper
Е
)regularization_losses
 Ѕlayer_regularization_losses
Іlayers
Їmetrics
Јnon_trainable_variables
*	variables
Љlayer_metrics
+trainable_variables
П__call__
+О&call_and_return_all_conditional_losses
'О"call_and_return_conditional_losses"
_generic_user_object
':%2conv2d/kernel
:2conv2d/bias
 "
trackable_list_wrapper
.
-0
.1"
trackable_list_wrapper
.
-0
.1"
trackable_list_wrapper
Е
/regularization_losses
 Њlayer_regularization_losses
Ћlayers
Ќmetrics
­non_trainable_variables
0	variables
Ўlayer_metrics
1trainable_variables
С__call__
+Р&call_and_return_all_conditional_losses
'Р"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
3regularization_losses
 Џlayer_regularization_losses
Аlayers
Бmetrics
Вnon_trainable_variables
4	variables
Гlayer_metrics
5trainable_variables
У__call__
+Т&call_and_return_all_conditional_losses
'Т"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):'2batch_normalization_1/gamma
(:&2batch_normalization_1/beta
1:/ (2!batch_normalization_1/moving_mean
5:3 (2%batch_normalization_1/moving_variance
 "
trackable_list_wrapper
<
80
91
:2
;3"
trackable_list_wrapper
.
80
91"
trackable_list_wrapper
Е
<regularization_losses
 Дlayer_regularization_losses
Еlayers
Жmetrics
Зnon_trainable_variables
=	variables
Иlayer_metrics
>trainable_variables
Х__call__
+Ф&call_and_return_all_conditional_losses
'Ф"call_and_return_conditional_losses"
_generic_user_object
):'2conv2d_1/kernel
:2conv2d_1/bias
 "
trackable_list_wrapper
.
@0
A1"
trackable_list_wrapper
.
@0
A1"
trackable_list_wrapper
Е
Bregularization_losses
 Йlayer_regularization_losses
Кlayers
Лmetrics
Мnon_trainable_variables
C	variables
Нlayer_metrics
Dtrainable_variables
Ч__call__
+Ц&call_and_return_all_conditional_losses
'Ц"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
Fregularization_losses
 Оlayer_regularization_losses
Пlayers
Рmetrics
Сnon_trainable_variables
G	variables
Тlayer_metrics
Htrainable_variables
Щ__call__
+Ш&call_and_return_all_conditional_losses
'Ш"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):'2batch_normalization_2/gamma
(:&2batch_normalization_2/beta
1:/ (2!batch_normalization_2/moving_mean
5:3 (2%batch_normalization_2/moving_variance
 "
trackable_list_wrapper
<
K0
L1
M2
N3"
trackable_list_wrapper
.
K0
L1"
trackable_list_wrapper
Е
Oregularization_losses
 Уlayer_regularization_losses
Фlayers
Хmetrics
Цnon_trainable_variables
P	variables
Чlayer_metrics
Qtrainable_variables
Ы__call__
+Ъ&call_and_return_all_conditional_losses
'Ъ"call_and_return_conditional_losses"
_generic_user_object
):' 2conv2d_2/kernel
: 2conv2d_2/bias
 "
trackable_list_wrapper
.
S0
T1"
trackable_list_wrapper
.
S0
T1"
trackable_list_wrapper
Е
Uregularization_losses
 Шlayer_regularization_losses
Щlayers
Ъmetrics
Ыnon_trainable_variables
V	variables
Ьlayer_metrics
Wtrainable_variables
Э__call__
+Ь&call_and_return_all_conditional_losses
'Ь"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
Yregularization_losses
 Эlayer_regularization_losses
Юlayers
Яmetrics
аnon_trainable_variables
Z	variables
бlayer_metrics
[trainable_variables
Я__call__
+Ю&call_and_return_all_conditional_losses
'Ю"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):' 2batch_normalization_3/gamma
(:& 2batch_normalization_3/beta
1:/  (2!batch_normalization_3/moving_mean
5:3  (2%batch_normalization_3/moving_variance
 "
trackable_list_wrapper
<
^0
_1
`2
a3"
trackable_list_wrapper
.
^0
_1"
trackable_list_wrapper
Е
bregularization_losses
 вlayer_regularization_losses
гlayers
дmetrics
еnon_trainable_variables
c	variables
жlayer_metrics
dtrainable_variables
б__call__
+а&call_and_return_all_conditional_losses
'а"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
fregularization_losses
 зlayer_regularization_losses
иlayers
йmetrics
кnon_trainable_variables
g	variables
лlayer_metrics
htrainable_variables
г__call__
+в&call_and_return_all_conditional_losses
'в"call_and_return_conditional_losses"
_generic_user_object
:	D@2dense/kernel
:@2
dense/bias
 "
trackable_list_wrapper
.
j0
k1"
trackable_list_wrapper
.
j0
k1"
trackable_list_wrapper
Е
lregularization_losses
 мlayer_regularization_losses
нlayers
оmetrics
пnon_trainable_variables
m	variables
рlayer_metrics
ntrainable_variables
е__call__
+д&call_and_return_all_conditional_losses
'д"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
pregularization_losses
 сlayer_regularization_losses
тlayers
уmetrics
фnon_trainable_variables
q	variables
хlayer_metrics
rtrainable_variables
з__call__
+ж&call_and_return_all_conditional_losses
'ж"call_and_return_conditional_losses"
_generic_user_object
 :@2dense_1/kernel
:2dense_1/bias
 "
trackable_list_wrapper
.
t0
u1"
trackable_list_wrapper
.
t0
u1"
trackable_list_wrapper
Е
vregularization_losses
 цlayer_regularization_losses
чlayers
шmetrics
щnon_trainable_variables
w	variables
ъlayer_metrics
xtrainable_variables
й__call__
+и&call_and_return_all_conditional_losses
'и"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
zregularization_losses
 ыlayer_regularization_losses
ьlayers
эmetrics
юnon_trainable_variables
{	variables
яlayer_metrics
|trainable_variables
л__call__
+к&call_and_return_all_conditional_losses
'к"call_and_return_conditional_losses"
_generic_user_object
 :2dense_2/kernel
:2dense_2/bias
 "
trackable_list_wrapper
.
~0
1"
trackable_list_wrapper
.
~0
1"
trackable_list_wrapper
И
regularization_losses
 №layer_regularization_losses
ёlayers
ђmetrics
ѓnon_trainable_variables
	variables
єlayer_metrics
trainable_variables
н__call__
+м&call_and_return_all_conditional_losses
'м"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_list_wrapper
І
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17"
trackable_list_wrapper
0
ѕ0
і1"
trackable_list_wrapper
X
'0
(1
:2
;3
M4
N5
`6
a7"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
regularization_losses
 їlayer_regularization_losses
јlayers
љmetrics
њnon_trainable_variables
	variables
ћlayer_metrics
trainable_variables
р__call__
+п&call_and_return_all_conditional_losses
'п"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
regularization_losses
 ќlayer_regularization_losses
§layers
ўmetrics
џnon_trainable_variables
	variables
layer_metrics
trainable_variables
т__call__
+с&call_and_return_all_conditional_losses
'с"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
regularization_losses
 layer_regularization_losses
layers
metrics
non_trainable_variables
	variables
layer_metrics
trainable_variables
ф__call__
+у&call_and_return_all_conditional_losses
'у"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
:0
;1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
M0
N1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
`0
a1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
П

total

count
	variables
	keras_api"
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
џ

total

count

_fn_kwargs
	variables
	keras_api"Г
_tf_keras_metric{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "binary_accuracy"}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
:  (2total
:  (2count
0
0
1"
trackable_list_wrapper
.
	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
0
1"
trackable_list_wrapper
.
	variables"
_generic_user_object
,:*2 Adam/batch_normalization/gamma/m
+:)2Adam/batch_normalization/beta/m
,:*2Adam/conv2d/kernel/m
:2Adam/conv2d/bias/m
.:,2"Adam/batch_normalization_1/gamma/m
-:+2!Adam/batch_normalization_1/beta/m
.:,2Adam/conv2d_1/kernel/m
 :2Adam/conv2d_1/bias/m
.:,2"Adam/batch_normalization_2/gamma/m
-:+2!Adam/batch_normalization_2/beta/m
.:, 2Adam/conv2d_2/kernel/m
 : 2Adam/conv2d_2/bias/m
.:, 2"Adam/batch_normalization_3/gamma/m
-:+ 2!Adam/batch_normalization_3/beta/m
$:"	D@2Adam/dense/kernel/m
:@2Adam/dense/bias/m
%:#@2Adam/dense_1/kernel/m
:2Adam/dense_1/bias/m
%:#2Adam/dense_2/kernel/m
:2Adam/dense_2/bias/m
,:*2 Adam/batch_normalization/gamma/v
+:)2Adam/batch_normalization/beta/v
,:*2Adam/conv2d/kernel/v
:2Adam/conv2d/bias/v
.:,2"Adam/batch_normalization_1/gamma/v
-:+2!Adam/batch_normalization_1/beta/v
.:,2Adam/conv2d_1/kernel/v
 :2Adam/conv2d_1/bias/v
.:,2"Adam/batch_normalization_2/gamma/v
-:+2!Adam/batch_normalization_2/beta/v
.:, 2Adam/conv2d_2/kernel/v
 : 2Adam/conv2d_2/bias/v
.:, 2"Adam/batch_normalization_3/gamma/v
-:+ 2!Adam/batch_normalization_3/beta/v
$:"	D@2Adam/dense/kernel/v
:@2Adam/dense/bias/v
%:#@2Adam/dense_1/kernel/v
:2Adam/dense_1/bias/v
%:#2Adam/dense_2/kernel/v
:2Adam/dense_2/bias/v
ц2у
 __inference__wrapped_model_19764О
В
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *.Ђ+
)&
reshape_inputџџџџџџџџџт	
т2п
E__inference_sequential_layer_call_and_return_conditional_losses_21007
E__inference_sequential_layer_call_and_return_conditional_losses_21717
E__inference_sequential_layer_call_and_return_conditional_losses_21965
E__inference_sequential_layer_call_and_return_conditional_losses_21087Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
і2ѓ
*__inference_sequential_layer_call_fn_22028
*__inference_sequential_layer_call_fn_22091
*__inference_sequential_layer_call_fn_21231
*__inference_sequential_layer_call_fn_21374Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ь2щ
B__inference_reshape_layer_call_and_return_conditional_losses_22104Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
б2Ю
'__inference_reshape_layer_call_fn_22109Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ђ2я
I__inference_melspectrogram_layer_call_and_return_conditional_losses_22243
I__inference_melspectrogram_layer_call_and_return_conditional_losses_22377
I__inference_melspectrogram_layer_call_and_return_conditional_losses_19948
I__inference_melspectrogram_layer_call_and_return_conditional_losses_19939Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
2
.__inference_melspectrogram_layer_call_fn_19965
.__inference_melspectrogram_layer_call_fn_22384
.__inference_melspectrogram_layer_call_fn_22391
.__inference_melspectrogram_layer_call_fn_19981Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
њ2ї
N__inference_batch_normalization_layer_call_and_return_conditional_losses_22411
N__inference_batch_normalization_layer_call_and_return_conditional_losses_22493
N__inference_batch_normalization_layer_call_and_return_conditional_losses_22429
N__inference_batch_normalization_layer_call_and_return_conditional_losses_22475Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
2
3__inference_batch_normalization_layer_call_fn_22455
3__inference_batch_normalization_layer_call_fn_22442
3__inference_batch_normalization_layer_call_fn_22506
3__inference_batch_normalization_layer_call_fn_22519Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ы2ш
A__inference_conv2d_layer_call_and_return_conditional_losses_22530Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
а2Э
&__inference_conv2d_layer_call_fn_22539Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
А2­
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_20091р
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *@Ђ=
;84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
2
-__inference_max_pooling2d_layer_call_fn_20097р
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *@Ђ=
;84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
2џ
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_22577
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_22559
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_22641
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_22623Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
2
5__inference_batch_normalization_1_layer_call_fn_22590
5__inference_batch_normalization_1_layer_call_fn_22603
5__inference_batch_normalization_1_layer_call_fn_22667
5__inference_batch_normalization_1_layer_call_fn_22654Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
э2ъ
C__inference_conv2d_1_layer_call_and_return_conditional_losses_22678Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
в2Я
(__inference_conv2d_1_layer_call_fn_22687Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
В2Џ
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_20207р
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *@Ђ=
;84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
2
/__inference_max_pooling2d_1_layer_call_fn_20213р
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *@Ђ=
;84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
2џ
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_22707
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_22771
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_22789
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_22725Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
2
5__inference_batch_normalization_2_layer_call_fn_22751
5__inference_batch_normalization_2_layer_call_fn_22815
5__inference_batch_normalization_2_layer_call_fn_22738
5__inference_batch_normalization_2_layer_call_fn_22802Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
э2ъ
C__inference_conv2d_2_layer_call_and_return_conditional_losses_22826Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
в2Я
(__inference_conv2d_2_layer_call_fn_22835Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
В2Џ
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_20323р
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *@Ђ=
;84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
2
/__inference_max_pooling2d_2_layer_call_fn_20329р
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *@Ђ=
;84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
2џ
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_22855
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_22919
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_22873
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_22937Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
2
5__inference_batch_normalization_3_layer_call_fn_22950
5__inference_batch_normalization_3_layer_call_fn_22963
5__inference_batch_normalization_3_layer_call_fn_22899
5__inference_batch_normalization_3_layer_call_fn_22886Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ь2щ
B__inference_flatten_layer_call_and_return_conditional_losses_22969Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
б2Ю
'__inference_flatten_layer_call_fn_22974Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ъ2ч
@__inference_dense_layer_call_and_return_conditional_losses_22985Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Я2Ь
%__inference_dense_layer_call_fn_22994Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Т2П
B__inference_dropout_layer_call_and_return_conditional_losses_23006
B__inference_dropout_layer_call_and_return_conditional_losses_23011Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
2
'__inference_dropout_layer_call_fn_23016
'__inference_dropout_layer_call_fn_23021Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ь2щ
B__inference_dense_1_layer_call_and_return_conditional_losses_23032Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
б2Ю
'__inference_dense_1_layer_call_fn_23041Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Ц2У
D__inference_dropout_1_layer_call_and_return_conditional_losses_23053
D__inference_dropout_1_layer_call_and_return_conditional_losses_23058Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
2
)__inference_dropout_1_layer_call_fn_23063
)__inference_dropout_1_layer_call_fn_23068Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ь2щ
B__inference_dense_2_layer_call_and_return_conditional_losses_23079Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
б2Ю
'__inference_dense_2_layer_call_fn_23088Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
аBЭ
#__inference_signature_wrapper_21447reshape_input"
В
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ф2с
?__inference_stft_layer_call_and_return_conditional_losses_23197
В
FullArgSpec
args
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Щ2Ц
$__inference_stft_layer_call_fn_23202
В
FullArgSpec
args
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
щ2ц
D__inference_magnitude_layer_call_and_return_conditional_losses_23207
В
FullArgSpec
args
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Ю2Ы
)__inference_magnitude_layer_call_fn_23212
В
FullArgSpec
args
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
№2э
K__inference_apply_filterbank_layer_call_and_return_conditional_losses_23240
В
FullArgSpec
args
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
е2в
0__inference_apply_filterbank_layer_call_fn_23247
В
FullArgSpec
args
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
	J
ConstВ
 __inference__wrapped_model_19764х%&'(-.89:;@AKLMNST^_`ajktu~8Ђ5
.Ђ+
)&
reshape_inputџџџџџџџџџт	
Њ "1Њ.
,
dense_2!
dense_2џџџџџџџџџК
K__inference_apply_filterbank_layer_call_and_return_conditional_losses_23240kх4Ђ1
*Ђ'
%"
xџџџџџџџџџЗ
Њ "/Ђ,
%"
0џџџџџџџџџЗ
 
0__inference_apply_filterbank_layer_call_fn_23247^х4Ђ1
*Ђ'
%"
xџџџџџџџџџЗ
Њ ""џџџџџџџџџЗы
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_2255989:;MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 ы
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_2257789:;MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p 
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Ц
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_22623r89:;;Ђ8
1Ђ.
(%
inputsџџџџџџџџџg@
p
Њ "-Ђ*
# 
0џџџџџџџџџg@
 Ц
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_22641r89:;;Ђ8
1Ђ.
(%
inputsџџџџџџџџџg@
p 
Њ "-Ђ*
# 
0џџџџџџџџџg@
 У
5__inference_batch_normalization_1_layer_call_fn_2259089:;MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџУ
5__inference_batch_normalization_1_layer_call_fn_2260389:;MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p 
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
5__inference_batch_normalization_1_layer_call_fn_22654e89:;;Ђ8
1Ђ.
(%
inputsџџџџџџџџџg@
p
Њ " џџџџџџџџџg@
5__inference_batch_normalization_1_layer_call_fn_22667e89:;;Ђ8
1Ђ.
(%
inputsџџџџџџџџџg@
p 
Њ " џџџџџџџџџg@ы
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_22707KLMNMЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 ы
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_22725KLMNMЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p 
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Ц
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_22771rKLMN;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ" 
p
Њ "-Ђ*
# 
0џџџџџџџџџ" 
 Ц
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_22789rKLMN;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ" 
p 
Њ "-Ђ*
# 
0џџџџџџџџџ" 
 У
5__inference_batch_normalization_2_layer_call_fn_22738KLMNMЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџУ
5__inference_batch_normalization_2_layer_call_fn_22751KLMNMЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p 
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
5__inference_batch_normalization_2_layer_call_fn_22802eKLMN;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ" 
p
Њ " џџџџџџџџџ" 
5__inference_batch_normalization_2_layer_call_fn_22815eKLMN;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ" 
p 
Њ " џџџџџџџџџ" ы
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_22855^_`aMЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
p
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 ы
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_22873^_`aMЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
p 
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 Ц
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_22919r^_`a;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ 
p
Њ "-Ђ*
# 
0џџџџџџџџџ 
 Ц
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_22937r^_`a;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ 
p 
Њ "-Ђ*
# 
0џџџџџџџџџ 
 У
5__inference_batch_normalization_3_layer_call_fn_22886^_`aMЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
p
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ У
5__inference_batch_normalization_3_layer_call_fn_22899^_`aMЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
p 
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
5__inference_batch_normalization_3_layer_call_fn_22950e^_`a;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ 
p
Њ " џџџџџџџџџ 
5__inference_batch_normalization_3_layer_call_fn_22963e^_`a;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ 
p 
Њ " џџџџџџџџџ щ
N__inference_batch_normalization_layer_call_and_return_conditional_losses_22411%&'(MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 щ
N__inference_batch_normalization_layer_call_and_return_conditional_losses_22429%&'(MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p 
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Ш
N__inference_batch_normalization_layer_call_and_return_conditional_losses_22475v%&'(=Ђ:
3Ђ0
*'
inputsџџџџџџџџџЗ
p
Њ "/Ђ,
%"
0џџџџџџџџџЗ
 Ш
N__inference_batch_normalization_layer_call_and_return_conditional_losses_22493v%&'(=Ђ:
3Ђ0
*'
inputsџџџџџџџџџЗ
p 
Њ "/Ђ,
%"
0џџџџџџџџџЗ
 С
3__inference_batch_normalization_layer_call_fn_22442%&'(MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџС
3__inference_batch_normalization_layer_call_fn_22455%&'(MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p 
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
3__inference_batch_normalization_layer_call_fn_22506i%&'(=Ђ:
3Ђ0
*'
inputsџџџџџџџџџЗ
p
Њ ""џџџџџџџџџЗ 
3__inference_batch_normalization_layer_call_fn_22519i%&'(=Ђ:
3Ђ0
*'
inputsџџџџџџџџџЗ
p 
Њ ""џџџџџџџџџЗГ
C__inference_conv2d_1_layer_call_and_return_conditional_losses_22678l@A7Ђ4
-Ђ*
(%
inputsџџџџџџџџџg@
Њ "-Ђ*
# 
0џџџџџџџџџg@
 
(__inference_conv2d_1_layer_call_fn_22687_@A7Ђ4
-Ђ*
(%
inputsџџџџџџџџџg@
Њ " џџџџџџџџџg@Г
C__inference_conv2d_2_layer_call_and_return_conditional_losses_22826lST7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ" 
Њ "-Ђ*
# 
0џџџџџџџџџ"  
 
(__inference_conv2d_2_layer_call_fn_22835_ST7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ" 
Њ " џџџџџџџџџ"  Е
A__inference_conv2d_layer_call_and_return_conditional_losses_22530p-.9Ђ6
/Ђ,
*'
inputsџџџџџџџџџЗ
Њ "/Ђ,
%"
0џџџџџџџџџЗ
 
&__inference_conv2d_layer_call_fn_22539c-.9Ђ6
/Ђ,
*'
inputsџџџџџџџџџЗ
Њ ""џџџџџџџџџЗЂ
B__inference_dense_1_layer_call_and_return_conditional_losses_23032\tu/Ђ,
%Ђ"
 
inputsџџџџџџџџџ@
Њ "%Ђ"

0џџџџџџџџџ
 z
'__inference_dense_1_layer_call_fn_23041Otu/Ђ,
%Ђ"
 
inputsџџџџџџџџџ@
Њ "џџџџџџџџџЂ
B__inference_dense_2_layer_call_and_return_conditional_losses_23079\~/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "%Ђ"

0џџџџџџџџџ
 z
'__inference_dense_2_layer_call_fn_23088O~/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "џџџџџџџџџЁ
@__inference_dense_layer_call_and_return_conditional_losses_22985]jk0Ђ-
&Ђ#
!
inputsџџџџџџџџџD
Њ "%Ђ"

0џџџџџџџџџ@
 y
%__inference_dense_layer_call_fn_22994Pjk0Ђ-
&Ђ#
!
inputsџџџџџџџџџD
Њ "џџџџџџџџџ@Є
D__inference_dropout_1_layer_call_and_return_conditional_losses_23053\3Ђ0
)Ђ&
 
inputsџџџџџџџџџ
p
Њ "%Ђ"

0џџџџџџџџџ
 Є
D__inference_dropout_1_layer_call_and_return_conditional_losses_23058\3Ђ0
)Ђ&
 
inputsџџџџџџџџџ
p 
Њ "%Ђ"

0џџџџџџџџџ
 |
)__inference_dropout_1_layer_call_fn_23063O3Ђ0
)Ђ&
 
inputsџџџџџџџџџ
p
Њ "џџџџџџџџџ|
)__inference_dropout_1_layer_call_fn_23068O3Ђ0
)Ђ&
 
inputsџџџџџџџџџ
p 
Њ "џџџџџџџџџЂ
B__inference_dropout_layer_call_and_return_conditional_losses_23006\3Ђ0
)Ђ&
 
inputsџџџџџџџџџ@
p
Њ "%Ђ"

0џџџџџџџџџ@
 Ђ
B__inference_dropout_layer_call_and_return_conditional_losses_23011\3Ђ0
)Ђ&
 
inputsџџџџџџџџџ@
p 
Њ "%Ђ"

0џџџџџџџџџ@
 z
'__inference_dropout_layer_call_fn_23016O3Ђ0
)Ђ&
 
inputsџџџџџџџџџ@
p
Њ "џџџџџџџџџ@z
'__inference_dropout_layer_call_fn_23021O3Ђ0
)Ђ&
 
inputsџџџџџџџџџ@
p 
Њ "џџџџџџџџџ@Ї
B__inference_flatten_layer_call_and_return_conditional_losses_22969a7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ 
Њ "&Ђ#

0џџџџџџџџџD
 
'__inference_flatten_layer_call_fn_22974T7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ 
Њ "џџџџџџџџџDЏ
D__inference_magnitude_layer_call_and_return_conditional_losses_23207g4Ђ1
*Ђ'
%"
xџџџџџџџџџЗ
Њ "/Ђ,
%"
0џџџџџџџџџЗ
 
)__inference_magnitude_layer_call_fn_23212Z4Ђ1
*Ђ'
%"
xџџџџџџџџџЗ
Њ ""џџџџџџџџџЗэ
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_20207RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "HЂE
>;
04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Х
/__inference_max_pooling2d_1_layer_call_fn_20213RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ ";84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџэ
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_20323RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "HЂE
>;
04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Х
/__inference_max_pooling2d_2_layer_call_fn_20329RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ ";84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџы
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_20091RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "HЂE
>;
04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 У
-__inference_max_pooling2d_layer_call_fn_20097RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ ";84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџХ
I__inference_melspectrogram_layer_call_and_return_conditional_losses_19939xхAЂ>
7Ђ4
*'

stft_inputџџџџџџџџџт	
p

 
Њ "/Ђ,
%"
0џџџџџџџџџЗ
 Х
I__inference_melspectrogram_layer_call_and_return_conditional_losses_19948xхAЂ>
7Ђ4
*'

stft_inputџџџџџџџџџт	
p 

 
Њ "/Ђ,
%"
0џџџџџџџџџЗ
 С
I__inference_melspectrogram_layer_call_and_return_conditional_losses_22243tх=Ђ:
3Ђ0
&#
inputsџџџџџџџџџт	
p

 
Њ "/Ђ,
%"
0џџџџџџџџџЗ
 С
I__inference_melspectrogram_layer_call_and_return_conditional_losses_22377tх=Ђ:
3Ђ0
&#
inputsџџџџџџџџџт	
p 

 
Њ "/Ђ,
%"
0џџџџџџџџџЗ
 
.__inference_melspectrogram_layer_call_fn_19965kхAЂ>
7Ђ4
*'

stft_inputџџџџџџџџџт	
p

 
Њ ""џџџџџџџџџЗ
.__inference_melspectrogram_layer_call_fn_19981kхAЂ>
7Ђ4
*'

stft_inputџџџџџџџџџт	
p 

 
Њ ""џџџџџџџџџЗ
.__inference_melspectrogram_layer_call_fn_22384gх=Ђ:
3Ђ0
&#
inputsџџџџџџџџџт	
p

 
Њ ""џџџџџџџџџЗ
.__inference_melspectrogram_layer_call_fn_22391gх=Ђ:
3Ђ0
&#
inputsџџџџџџџџџт	
p 

 
Њ ""џџџџџџџџџЗІ
B__inference_reshape_layer_call_and_return_conditional_losses_22104`1Ђ.
'Ђ$
"
inputsџџџџџџџџџт	
Њ "+Ђ(
!
0џџџџџџџџџт	
 ~
'__inference_reshape_layer_call_fn_22109S1Ђ.
'Ђ$
"
inputsџџџџџџџџџт	
Њ "џџџџџџџџџт	г
E__inference_sequential_layer_call_and_return_conditional_losses_21007х%&'(-.89:;@AKLMNST^_`ajktu~@Ђ=
6Ђ3
)&
reshape_inputџџџџџџџџџт	
p

 
Њ "%Ђ"

0џџџџџџџџџ
 г
E__inference_sequential_layer_call_and_return_conditional_losses_21087х%&'(-.89:;@AKLMNST^_`ajktu~@Ђ=
6Ђ3
)&
reshape_inputџџџџџџџџџт	
p 

 
Њ "%Ђ"

0џџџџџџџџџ
 Ь
E__inference_sequential_layer_call_and_return_conditional_losses_21717х%&'(-.89:;@AKLMNST^_`ajktu~9Ђ6
/Ђ,
"
inputsџџџџџџџџџт	
p

 
Њ "%Ђ"

0џџџџџџџџџ
 Ь
E__inference_sequential_layer_call_and_return_conditional_losses_21965х%&'(-.89:;@AKLMNST^_`ajktu~9Ђ6
/Ђ,
"
inputsџџџџџџџџџт	
p 

 
Њ "%Ђ"

0џџџџџџџџџ
 Њ
*__inference_sequential_layer_call_fn_21231|х%&'(-.89:;@AKLMNST^_`ajktu~@Ђ=
6Ђ3
)&
reshape_inputџџџџџџџџџт	
p

 
Њ "џџџџџџџџџЊ
*__inference_sequential_layer_call_fn_21374|х%&'(-.89:;@AKLMNST^_`ajktu~@Ђ=
6Ђ3
)&
reshape_inputџџџџџџџџџт	
p 

 
Њ "џџџџџџџџџЃ
*__inference_sequential_layer_call_fn_22028uх%&'(-.89:;@AKLMNST^_`ajktu~9Ђ6
/Ђ,
"
inputsџџџџџџџџџт	
p

 
Њ "џџџџџџџџџЃ
*__inference_sequential_layer_call_fn_22091uх%&'(-.89:;@AKLMNST^_`ajktu~9Ђ6
/Ђ,
"
inputsџџџџџџџџџт	
p 

 
Њ "џџџџџџџџџЦ
#__inference_signature_wrapper_21447х%&'(-.89:;@AKLMNST^_`ajktu~IЂF
Ђ 
?Њ<
:
reshape_input)&
reshape_inputџџџџџџџџџт	"1Њ.
,
dense_2!
dense_2џџџџџџџџџІ
?__inference_stft_layer_call_and_return_conditional_losses_23197c0Ђ-
&Ђ#
!
xџџџџџџџџџт	
Њ "/Ђ,
%"
0џџџџџџџџџЗ
 ~
$__inference_stft_layer_call_fn_23202V0Ђ-
&Ђ#
!
xџџџџџџџџџт	
Њ ""џџџџџџџџџЗ