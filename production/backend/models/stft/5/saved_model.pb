┐т&
ф#╣#
B
AddV2
x"T
y"T
z"T"
Ttype:
2	АР
B
AssignVariableOp
resource
value"dtype"
dtypetypeИ
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
Ы
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
·
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
epsilonfloat%╖╤8"&
exponential_avg_factorfloat%  А?";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
н
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
В
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
delete_old_dirsbool(И
=
Mul
x"T
y"T
z"T"
Ttype:
2	Р
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
dtypetypeИ
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
list(type)(0И
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
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
Л
SplitV

value"T
size_splits"Tlen
	split_dim
output"T*	num_split"
	num_splitint(0"	
Ttype"
Tlentype0	:
2	
╛
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
executor_typestring И
@
StaticRegexFullMatch	
input

output
"
patternstring
Ў
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
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И"serve*2.4.12v2.4.0-49-g85c8b2a817f8пп!
О
batch_normalization_4/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_4/gamma
З
/batch_normalization_4/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_4/gamma*
_output_shapes
:*
dtype0
М
batch_normalization_4/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namebatch_normalization_4/beta
Е
.batch_normalization_4/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_4/beta*
_output_shapes
:*
dtype0
Ъ
!batch_normalization_4/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!batch_normalization_4/moving_mean
У
5batch_normalization_4/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_4/moving_mean*
_output_shapes
:*
dtype0
в
%batch_normalization_4/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%batch_normalization_4/moving_variance
Ы
9batch_normalization_4/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_4/moving_variance*
_output_shapes
:*
dtype0
В
conv2d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_3/kernel
{
#conv2d_3/kernel/Read/ReadVariableOpReadVariableOpconv2d_3/kernel*&
_output_shapes
:*
dtype0
r
conv2d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_3/bias
k
!conv2d_3/bias/Read/ReadVariableOpReadVariableOpconv2d_3/bias*
_output_shapes
:*
dtype0
О
batch_normalization_5/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_5/gamma
З
/batch_normalization_5/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_5/gamma*
_output_shapes
:*
dtype0
М
batch_normalization_5/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namebatch_normalization_5/beta
Е
.batch_normalization_5/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_5/beta*
_output_shapes
:*
dtype0
Ъ
!batch_normalization_5/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!batch_normalization_5/moving_mean
У
5batch_normalization_5/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_5/moving_mean*
_output_shapes
:*
dtype0
в
%batch_normalization_5/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%batch_normalization_5/moving_variance
Ы
9batch_normalization_5/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_5/moving_variance*
_output_shapes
:*
dtype0
В
conv2d_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_4/kernel
{
#conv2d_4/kernel/Read/ReadVariableOpReadVariableOpconv2d_4/kernel*&
_output_shapes
:*
dtype0
r
conv2d_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_4/bias
k
!conv2d_4/bias/Read/ReadVariableOpReadVariableOpconv2d_4/bias*
_output_shapes
:*
dtype0
О
batch_normalization_6/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_6/gamma
З
/batch_normalization_6/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_6/gamma*
_output_shapes
:*
dtype0
М
batch_normalization_6/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namebatch_normalization_6/beta
Е
.batch_normalization_6/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_6/beta*
_output_shapes
:*
dtype0
Ъ
!batch_normalization_6/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!batch_normalization_6/moving_mean
У
5batch_normalization_6/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_6/moving_mean*
_output_shapes
:*
dtype0
в
%batch_normalization_6/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%batch_normalization_6/moving_variance
Ы
9batch_normalization_6/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_6/moving_variance*
_output_shapes
:*
dtype0
В
conv2d_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nameconv2d_5/kernel
{
#conv2d_5/kernel/Read/ReadVariableOpReadVariableOpconv2d_5/kernel*&
_output_shapes
: *
dtype0
r
conv2d_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_5/bias
k
!conv2d_5/bias/Read/ReadVariableOpReadVariableOpconv2d_5/bias*
_output_shapes
: *
dtype0
О
batch_normalization_7/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_7/gamma
З
/batch_normalization_7/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_7/gamma*
_output_shapes
: *
dtype0
М
batch_normalization_7/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_namebatch_normalization_7/beta
Е
.batch_normalization_7/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_7/beta*
_output_shapes
: *
dtype0
Ъ
!batch_normalization_7/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!batch_normalization_7/moving_mean
У
5batch_normalization_7/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_7/moving_mean*
_output_shapes
: *
dtype0
в
%batch_normalization_7/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *6
shared_name'%batch_normalization_7/moving_variance
Ы
9batch_normalization_7/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_7/moving_variance*
_output_shapes
: *
dtype0
y
dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	рv@*
shared_namedense_3/kernel
r
"dense_3/kernel/Read/ReadVariableOpReadVariableOpdense_3/kernel*
_output_shapes
:	рv@*
dtype0
p
dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_3/bias
i
 dense_3/bias/Read/ReadVariableOpReadVariableOpdense_3/bias*
_output_shapes
:@*
dtype0
x
dense_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*
shared_namedense_4/kernel
q
"dense_4/kernel/Read/ReadVariableOpReadVariableOpdense_4/kernel*
_output_shapes

:@*
dtype0
p
dense_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_4/bias
i
 dense_4/bias/Read/ReadVariableOpReadVariableOpdense_4/bias*
_output_shapes
:*
dtype0
x
dense_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense_5/kernel
q
"dense_5/kernel/Read/ReadVariableOpReadVariableOpdense_5/kernel*
_output_shapes

:*
dtype0
p
dense_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_5/bias
i
 dense_5/bias/Read/ReadVariableOpReadVariableOpdense_5/bias*
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
Ь
"Adam/batch_normalization_4/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_4/gamma/m
Х
6Adam/batch_normalization_4/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_4/gamma/m*
_output_shapes
:*
dtype0
Ъ
!Adam/batch_normalization_4/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/batch_normalization_4/beta/m
У
5Adam/batch_normalization_4/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_4/beta/m*
_output_shapes
:*
dtype0
Р
Adam/conv2d_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_3/kernel/m
Й
*Adam/conv2d_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_3/kernel/m*&
_output_shapes
:*
dtype0
А
Adam/conv2d_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d_3/bias/m
y
(Adam/conv2d_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_3/bias/m*
_output_shapes
:*
dtype0
Ь
"Adam/batch_normalization_5/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_5/gamma/m
Х
6Adam/batch_normalization_5/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_5/gamma/m*
_output_shapes
:*
dtype0
Ъ
!Adam/batch_normalization_5/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/batch_normalization_5/beta/m
У
5Adam/batch_normalization_5/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_5/beta/m*
_output_shapes
:*
dtype0
Р
Adam/conv2d_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_4/kernel/m
Й
*Adam/conv2d_4/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_4/kernel/m*&
_output_shapes
:*
dtype0
А
Adam/conv2d_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d_4/bias/m
y
(Adam/conv2d_4/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_4/bias/m*
_output_shapes
:*
dtype0
Ь
"Adam/batch_normalization_6/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_6/gamma/m
Х
6Adam/batch_normalization_6/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_6/gamma/m*
_output_shapes
:*
dtype0
Ъ
!Adam/batch_normalization_6/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/batch_normalization_6/beta/m
У
5Adam/batch_normalization_6/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_6/beta/m*
_output_shapes
:*
dtype0
Р
Adam/conv2d_5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdam/conv2d_5/kernel/m
Й
*Adam/conv2d_5/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_5/kernel/m*&
_output_shapes
: *
dtype0
А
Adam/conv2d_5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/conv2d_5/bias/m
y
(Adam/conv2d_5/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_5/bias/m*
_output_shapes
: *
dtype0
Ь
"Adam/batch_normalization_7/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"Adam/batch_normalization_7/gamma/m
Х
6Adam/batch_normalization_7/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_7/gamma/m*
_output_shapes
: *
dtype0
Ъ
!Adam/batch_normalization_7/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!Adam/batch_normalization_7/beta/m
У
5Adam/batch_normalization_7/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_7/beta/m*
_output_shapes
: *
dtype0
З
Adam/dense_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	рv@*&
shared_nameAdam/dense_3/kernel/m
А
)Adam/dense_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_3/kernel/m*
_output_shapes
:	рv@*
dtype0
~
Adam/dense_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameAdam/dense_3/bias/m
w
'Adam/dense_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_3/bias/m*
_output_shapes
:@*
dtype0
Ж
Adam/dense_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*&
shared_nameAdam/dense_4/kernel/m

)Adam/dense_4/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_4/kernel/m*
_output_shapes

:@*
dtype0
~
Adam/dense_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_4/bias/m
w
'Adam/dense_4/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_4/bias/m*
_output_shapes
:*
dtype0
Ж
Adam/dense_5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameAdam/dense_5/kernel/m

)Adam/dense_5/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_5/kernel/m*
_output_shapes

:*
dtype0
~
Adam/dense_5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_5/bias/m
w
'Adam/dense_5/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_5/bias/m*
_output_shapes
:*
dtype0
Ь
"Adam/batch_normalization_4/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_4/gamma/v
Х
6Adam/batch_normalization_4/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_4/gamma/v*
_output_shapes
:*
dtype0
Ъ
!Adam/batch_normalization_4/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/batch_normalization_4/beta/v
У
5Adam/batch_normalization_4/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_4/beta/v*
_output_shapes
:*
dtype0
Р
Adam/conv2d_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_3/kernel/v
Й
*Adam/conv2d_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_3/kernel/v*&
_output_shapes
:*
dtype0
А
Adam/conv2d_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d_3/bias/v
y
(Adam/conv2d_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_3/bias/v*
_output_shapes
:*
dtype0
Ь
"Adam/batch_normalization_5/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_5/gamma/v
Х
6Adam/batch_normalization_5/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_5/gamma/v*
_output_shapes
:*
dtype0
Ъ
!Adam/batch_normalization_5/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/batch_normalization_5/beta/v
У
5Adam/batch_normalization_5/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_5/beta/v*
_output_shapes
:*
dtype0
Р
Adam/conv2d_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_4/kernel/v
Й
*Adam/conv2d_4/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_4/kernel/v*&
_output_shapes
:*
dtype0
А
Adam/conv2d_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d_4/bias/v
y
(Adam/conv2d_4/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_4/bias/v*
_output_shapes
:*
dtype0
Ь
"Adam/batch_normalization_6/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_6/gamma/v
Х
6Adam/batch_normalization_6/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_6/gamma/v*
_output_shapes
:*
dtype0
Ъ
!Adam/batch_normalization_6/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/batch_normalization_6/beta/v
У
5Adam/batch_normalization_6/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_6/beta/v*
_output_shapes
:*
dtype0
Р
Adam/conv2d_5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdam/conv2d_5/kernel/v
Й
*Adam/conv2d_5/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_5/kernel/v*&
_output_shapes
: *
dtype0
А
Adam/conv2d_5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/conv2d_5/bias/v
y
(Adam/conv2d_5/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_5/bias/v*
_output_shapes
: *
dtype0
Ь
"Adam/batch_normalization_7/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"Adam/batch_normalization_7/gamma/v
Х
6Adam/batch_normalization_7/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_7/gamma/v*
_output_shapes
: *
dtype0
Ъ
!Adam/batch_normalization_7/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!Adam/batch_normalization_7/beta/v
У
5Adam/batch_normalization_7/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_7/beta/v*
_output_shapes
: *
dtype0
З
Adam/dense_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	рv@*&
shared_nameAdam/dense_3/kernel/v
А
)Adam/dense_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_3/kernel/v*
_output_shapes
:	рv@*
dtype0
~
Adam/dense_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameAdam/dense_3/bias/v
w
'Adam/dense_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_3/bias/v*
_output_shapes
:@*
dtype0
Ж
Adam/dense_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*&
shared_nameAdam/dense_4/kernel/v

)Adam/dense_4/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_4/kernel/v*
_output_shapes

:@*
dtype0
~
Adam/dense_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_4/bias/v
w
'Adam/dense_4/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_4/bias/v*
_output_shapes
:*
dtype0
Ж
Adam/dense_5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameAdam/dense_5/kernel/v

)Adam/dense_5/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_5/kernel/v*
_output_shapes

:*
dtype0
~
Adam/dense_5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_5/bias/v
w
'Adam/dense_5/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_5/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
▌И
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ЧИ
valueМИBИИ BАИ
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
	variables
trainable_variables
regularization_losses
	keras_api

signatures
R
	variables
trainable_variables
regularization_losses
	keras_api
l
layer-0
layer-1
	variables
 trainable_variables
!regularization_losses
"	keras_api
Ч
#axis
	$gamma
%beta
&moving_mean
'moving_variance
(	variables
)trainable_variables
*regularization_losses
+	keras_api
h

,kernel
-bias
.	variables
/trainable_variables
0regularization_losses
1	keras_api
R
2	variables
3trainable_variables
4regularization_losses
5	keras_api
Ч
6axis
	7gamma
8beta
9moving_mean
:moving_variance
;	variables
<trainable_variables
=regularization_losses
>	keras_api
h

?kernel
@bias
A	variables
Btrainable_variables
Cregularization_losses
D	keras_api
R
E	variables
Ftrainable_variables
Gregularization_losses
H	keras_api
Ч
Iaxis
	Jgamma
Kbeta
Lmoving_mean
Mmoving_variance
N	variables
Otrainable_variables
Pregularization_losses
Q	keras_api
h

Rkernel
Sbias
T	variables
Utrainable_variables
Vregularization_losses
W	keras_api
R
X	variables
Ytrainable_variables
Zregularization_losses
[	keras_api
Ч
\axis
	]gamma
^beta
_moving_mean
`moving_variance
a	variables
btrainable_variables
cregularization_losses
d	keras_api
R
e	variables
ftrainable_variables
gregularization_losses
h	keras_api
h

ikernel
jbias
k	variables
ltrainable_variables
mregularization_losses
n	keras_api
R
o	variables
ptrainable_variables
qregularization_losses
r	keras_api
h

skernel
tbias
u	variables
vtrainable_variables
wregularization_losses
x	keras_api
R
y	variables
ztrainable_variables
{regularization_losses
|	keras_api
k

}kernel
~bias
	variables
Аtrainable_variables
Бregularization_losses
В	keras_api
╒
	Гiter
Дbeta_1
Еbeta_2

Жdecay
Зlearning_rate$mД%mЕ,mЖ-mЗ7mИ8mЙ?mК@mЛJmМKmНRmОSmП]mР^mСimТjmУsmФtmХ}mЦ~mЧ$vШ%vЩ,vЪ-vЫ7vЬ8vЭ?vЮ@vЯJvаKvбRvвSvг]vд^vеivжjvзsvиtvй}vк~vл
╓
$0
%1
&2
'3
,4
-5
76
87
98
:9
?10
@11
J12
K13
L14
M15
R16
S17
]18
^19
_20
`21
i22
j23
s24
t25
}26
~27
Ц
$0
%1
,2
-3
74
85
?6
@7
J8
K9
R10
S11
]12
^13
i14
j15
s16
t17
}18
~19
 
▓
 Иlayer_regularization_losses
	variables
Йmetrics
Кnon_trainable_variables
trainable_variables
Лlayer_metrics
regularization_losses
Мlayers
 
 
 
 
▓
 Нlayer_regularization_losses
Оmetrics
Пnon_trainable_variables
	variables
trainable_variables
Рlayer_metrics
regularization_losses
Сlayers
V
Т	variables
Уtrainable_variables
Фregularization_losses
Х	keras_api
V
Ц	variables
Чtrainable_variables
Шregularization_losses
Щ	keras_api
 
 
 
▓
 Ъlayer_regularization_losses
	variables
Ыmetrics
Ьnon_trainable_variables
 trainable_variables
Эlayer_metrics
!regularization_losses
Юlayers
 
fd
VARIABLE_VALUEbatch_normalization_4/gamma5layer_with_weights-0/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_4/beta4layer_with_weights-0/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_4/moving_mean;layer_with_weights-0/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_4/moving_variance?layer_with_weights-0/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

$0
%1
&2
'3

$0
%1
 
▓
 Яlayer_regularization_losses
аmetrics
бnon_trainable_variables
(	variables
)trainable_variables
вlayer_metrics
*regularization_losses
гlayers
[Y
VARIABLE_VALUEconv2d_3/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_3/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

,0
-1

,0
-1
 
▓
 дlayer_regularization_losses
еmetrics
жnon_trainable_variables
.	variables
/trainable_variables
зlayer_metrics
0regularization_losses
иlayers
 
 
 
▓
 йlayer_regularization_losses
кmetrics
лnon_trainable_variables
2	variables
3trainable_variables
мlayer_metrics
4regularization_losses
нlayers
 
fd
VARIABLE_VALUEbatch_normalization_5/gamma5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_5/beta4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_5/moving_mean;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_5/moving_variance?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

70
81
92
:3

70
81
 
▓
 оlayer_regularization_losses
пmetrics
░non_trainable_variables
;	variables
<trainable_variables
▒layer_metrics
=regularization_losses
▓layers
[Y
VARIABLE_VALUEconv2d_4/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_4/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

?0
@1

?0
@1
 
▓
 │layer_regularization_losses
┤metrics
╡non_trainable_variables
A	variables
Btrainable_variables
╢layer_metrics
Cregularization_losses
╖layers
 
 
 
▓
 ╕layer_regularization_losses
╣metrics
║non_trainable_variables
E	variables
Ftrainable_variables
╗layer_metrics
Gregularization_losses
╝layers
 
fd
VARIABLE_VALUEbatch_normalization_6/gamma5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_6/beta4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_6/moving_mean;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_6/moving_variance?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

J0
K1
L2
M3

J0
K1
 
▓
 ╜layer_regularization_losses
╛metrics
┐non_trainable_variables
N	variables
Otrainable_variables
└layer_metrics
Pregularization_losses
┴layers
[Y
VARIABLE_VALUEconv2d_5/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_5/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

R0
S1

R0
S1
 
▓
 ┬layer_regularization_losses
├metrics
─non_trainable_variables
T	variables
Utrainable_variables
┼layer_metrics
Vregularization_losses
╞layers
 
 
 
▓
 ╟layer_regularization_losses
╚metrics
╔non_trainable_variables
X	variables
Ytrainable_variables
╩layer_metrics
Zregularization_losses
╦layers
 
fd
VARIABLE_VALUEbatch_normalization_7/gamma5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_7/beta4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_7/moving_mean;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_7/moving_variance?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

]0
^1
_2
`3

]0
^1
 
▓
 ╠layer_regularization_losses
═metrics
╬non_trainable_variables
a	variables
btrainable_variables
╧layer_metrics
cregularization_losses
╨layers
 
 
 
▓
 ╤layer_regularization_losses
╥metrics
╙non_trainable_variables
e	variables
ftrainable_variables
╘layer_metrics
gregularization_losses
╒layers
ZX
VARIABLE_VALUEdense_3/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_3/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE

i0
j1

i0
j1
 
▓
 ╓layer_regularization_losses
╫metrics
╪non_trainable_variables
k	variables
ltrainable_variables
┘layer_metrics
mregularization_losses
┌layers
 
 
 
▓
 █layer_regularization_losses
▄metrics
▌non_trainable_variables
o	variables
ptrainable_variables
▐layer_metrics
qregularization_losses
▀layers
ZX
VARIABLE_VALUEdense_4/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_4/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE

s0
t1

s0
t1
 
▓
 рlayer_regularization_losses
сmetrics
тnon_trainable_variables
u	variables
vtrainable_variables
уlayer_metrics
wregularization_losses
фlayers
 
 
 
▓
 хlayer_regularization_losses
цmetrics
чnon_trainable_variables
y	variables
ztrainable_variables
шlayer_metrics
{regularization_losses
щlayers
ZX
VARIABLE_VALUEdense_5/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_5/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE

}0
~1

}0
~1
 
┤
 ъlayer_regularization_losses
ыmetrics
ьnon_trainable_variables
	variables
Аtrainable_variables
эlayer_metrics
Бregularization_losses
юlayers
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

я0
Ё1
8
&0
'1
92
:3
L4
M5
_6
`7
 
Ж
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
 
 
 
 
 
 
 
 
╡
 ёlayer_regularization_losses
Єmetrics
єnon_trainable_variables
Т	variables
Уtrainable_variables
Їlayer_metrics
Фregularization_losses
їlayers
 
 
 
╡
 Ўlayer_regularization_losses
ўmetrics
°non_trainable_variables
Ц	variables
Чtrainable_variables
∙layer_metrics
Шregularization_losses
·layers
 
 
 
 

0
1
 
 

&0
'1
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
90
:1
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
L0
M1
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
_0
`1
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
 
8

√total

№count
¤	variables
■	keras_api
I

 total

Аcount
Б
_fn_kwargs
В	variables
Г	keras_api
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
√0
№1

¤	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

 0
А1

В	variables
КЗ
VARIABLE_VALUE"Adam/batch_normalization_4/gamma/mQlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ИЕ
VARIABLE_VALUE!Adam/batch_normalization_4/beta/mPlayer_with_weights-0/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_3/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_3/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
КЗ
VARIABLE_VALUE"Adam/batch_normalization_5/gamma/mQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ИЕ
VARIABLE_VALUE!Adam/batch_normalization_5/beta/mPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_4/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_4/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
КЗ
VARIABLE_VALUE"Adam/batch_normalization_6/gamma/mQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ИЕ
VARIABLE_VALUE!Adam/batch_normalization_6/beta/mPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_5/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_5/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
КЗ
VARIABLE_VALUE"Adam/batch_normalization_7/gamma/mQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ИЕ
VARIABLE_VALUE!Adam/batch_normalization_7/beta/mPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_3/kernel/mRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_3/bias/mPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_4/kernel/mRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_4/bias/mPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_5/kernel/mRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_5/bias/mPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
КЗ
VARIABLE_VALUE"Adam/batch_normalization_4/gamma/vQlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ИЕ
VARIABLE_VALUE!Adam/batch_normalization_4/beta/vPlayer_with_weights-0/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_3/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_3/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
КЗ
VARIABLE_VALUE"Adam/batch_normalization_5/gamma/vQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ИЕ
VARIABLE_VALUE!Adam/batch_normalization_5/beta/vPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_4/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_4/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
КЗ
VARIABLE_VALUE"Adam/batch_normalization_6/gamma/vQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ИЕ
VARIABLE_VALUE!Adam/batch_normalization_6/beta/vPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_5/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_5/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
КЗ
VARIABLE_VALUE"Adam/batch_normalization_7/gamma/vQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ИЕ
VARIABLE_VALUE!Adam/batch_normalization_7/beta/vPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_3/kernel/vRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_3/bias/vPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_4/kernel/vRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_4/bias/vPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_5/kernel/vRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_5/bias/vPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
Ж
serving_default_reshape_1_inputPlaceholder*)
_output_shapes
:         Ат	*
dtype0*
shape:         Ат	
╖
StatefulPartitionedCallStatefulPartitionedCallserving_default_reshape_1_inputbatch_normalization_4/gammabatch_normalization_4/beta!batch_normalization_4/moving_mean%batch_normalization_4/moving_varianceconv2d_3/kernelconv2d_3/biasbatch_normalization_5/gammabatch_normalization_5/beta!batch_normalization_5/moving_mean%batch_normalization_5/moving_varianceconv2d_4/kernelconv2d_4/biasbatch_normalization_6/gammabatch_normalization_6/beta!batch_normalization_6/moving_mean%batch_normalization_6/moving_varianceconv2d_5/kernelconv2d_5/biasbatch_normalization_7/gammabatch_normalization_7/beta!batch_normalization_7/moving_mean%batch_normalization_7/moving_variancedense_3/kerneldense_3/biasdense_4/kerneldense_4/biasdense_5/kerneldense_5/bias*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *>
_read_only_resource_inputs 
	
*-
config_proto

CPU

GPU 2J 8В *,
f'R%
#__inference_signature_wrapper_45210
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ъ
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename/batch_normalization_4/gamma/Read/ReadVariableOp.batch_normalization_4/beta/Read/ReadVariableOp5batch_normalization_4/moving_mean/Read/ReadVariableOp9batch_normalization_4/moving_variance/Read/ReadVariableOp#conv2d_3/kernel/Read/ReadVariableOp!conv2d_3/bias/Read/ReadVariableOp/batch_normalization_5/gamma/Read/ReadVariableOp.batch_normalization_5/beta/Read/ReadVariableOp5batch_normalization_5/moving_mean/Read/ReadVariableOp9batch_normalization_5/moving_variance/Read/ReadVariableOp#conv2d_4/kernel/Read/ReadVariableOp!conv2d_4/bias/Read/ReadVariableOp/batch_normalization_6/gamma/Read/ReadVariableOp.batch_normalization_6/beta/Read/ReadVariableOp5batch_normalization_6/moving_mean/Read/ReadVariableOp9batch_normalization_6/moving_variance/Read/ReadVariableOp#conv2d_5/kernel/Read/ReadVariableOp!conv2d_5/bias/Read/ReadVariableOp/batch_normalization_7/gamma/Read/ReadVariableOp.batch_normalization_7/beta/Read/ReadVariableOp5batch_normalization_7/moving_mean/Read/ReadVariableOp9batch_normalization_7/moving_variance/Read/ReadVariableOp"dense_3/kernel/Read/ReadVariableOp dense_3/bias/Read/ReadVariableOp"dense_4/kernel/Read/ReadVariableOp dense_4/bias/Read/ReadVariableOp"dense_5/kernel/Read/ReadVariableOp dense_5/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp6Adam/batch_normalization_4/gamma/m/Read/ReadVariableOp5Adam/batch_normalization_4/beta/m/Read/ReadVariableOp*Adam/conv2d_3/kernel/m/Read/ReadVariableOp(Adam/conv2d_3/bias/m/Read/ReadVariableOp6Adam/batch_normalization_5/gamma/m/Read/ReadVariableOp5Adam/batch_normalization_5/beta/m/Read/ReadVariableOp*Adam/conv2d_4/kernel/m/Read/ReadVariableOp(Adam/conv2d_4/bias/m/Read/ReadVariableOp6Adam/batch_normalization_6/gamma/m/Read/ReadVariableOp5Adam/batch_normalization_6/beta/m/Read/ReadVariableOp*Adam/conv2d_5/kernel/m/Read/ReadVariableOp(Adam/conv2d_5/bias/m/Read/ReadVariableOp6Adam/batch_normalization_7/gamma/m/Read/ReadVariableOp5Adam/batch_normalization_7/beta/m/Read/ReadVariableOp)Adam/dense_3/kernel/m/Read/ReadVariableOp'Adam/dense_3/bias/m/Read/ReadVariableOp)Adam/dense_4/kernel/m/Read/ReadVariableOp'Adam/dense_4/bias/m/Read/ReadVariableOp)Adam/dense_5/kernel/m/Read/ReadVariableOp'Adam/dense_5/bias/m/Read/ReadVariableOp6Adam/batch_normalization_4/gamma/v/Read/ReadVariableOp5Adam/batch_normalization_4/beta/v/Read/ReadVariableOp*Adam/conv2d_3/kernel/v/Read/ReadVariableOp(Adam/conv2d_3/bias/v/Read/ReadVariableOp6Adam/batch_normalization_5/gamma/v/Read/ReadVariableOp5Adam/batch_normalization_5/beta/v/Read/ReadVariableOp*Adam/conv2d_4/kernel/v/Read/ReadVariableOp(Adam/conv2d_4/bias/v/Read/ReadVariableOp6Adam/batch_normalization_6/gamma/v/Read/ReadVariableOp5Adam/batch_normalization_6/beta/v/Read/ReadVariableOp*Adam/conv2d_5/kernel/v/Read/ReadVariableOp(Adam/conv2d_5/bias/v/Read/ReadVariableOp6Adam/batch_normalization_7/gamma/v/Read/ReadVariableOp5Adam/batch_normalization_7/beta/v/Read/ReadVariableOp)Adam/dense_3/kernel/v/Read/ReadVariableOp'Adam/dense_3/bias/v/Read/ReadVariableOp)Adam/dense_4/kernel/v/Read/ReadVariableOp'Adam/dense_4/bias/v/Read/ReadVariableOp)Adam/dense_5/kernel/v/Read/ReadVariableOp'Adam/dense_5/bias/v/Read/ReadVariableOpConst*Z
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
GPU 2J 8В *'
f"R 
__inference__traced_save_47125
с
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamebatch_normalization_4/gammabatch_normalization_4/beta!batch_normalization_4/moving_mean%batch_normalization_4/moving_varianceconv2d_3/kernelconv2d_3/biasbatch_normalization_5/gammabatch_normalization_5/beta!batch_normalization_5/moving_mean%batch_normalization_5/moving_varianceconv2d_4/kernelconv2d_4/biasbatch_normalization_6/gammabatch_normalization_6/beta!batch_normalization_6/moving_mean%batch_normalization_6/moving_varianceconv2d_5/kernelconv2d_5/biasbatch_normalization_7/gammabatch_normalization_7/beta!batch_normalization_7/moving_mean%batch_normalization_7/moving_variancedense_3/kerneldense_3/biasdense_4/kerneldense_4/biasdense_5/kerneldense_5/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1"Adam/batch_normalization_4/gamma/m!Adam/batch_normalization_4/beta/mAdam/conv2d_3/kernel/mAdam/conv2d_3/bias/m"Adam/batch_normalization_5/gamma/m!Adam/batch_normalization_5/beta/mAdam/conv2d_4/kernel/mAdam/conv2d_4/bias/m"Adam/batch_normalization_6/gamma/m!Adam/batch_normalization_6/beta/mAdam/conv2d_5/kernel/mAdam/conv2d_5/bias/m"Adam/batch_normalization_7/gamma/m!Adam/batch_normalization_7/beta/mAdam/dense_3/kernel/mAdam/dense_3/bias/mAdam/dense_4/kernel/mAdam/dense_4/bias/mAdam/dense_5/kernel/mAdam/dense_5/bias/m"Adam/batch_normalization_4/gamma/v!Adam/batch_normalization_4/beta/vAdam/conv2d_3/kernel/vAdam/conv2d_3/bias/v"Adam/batch_normalization_5/gamma/v!Adam/batch_normalization_5/beta/vAdam/conv2d_4/kernel/vAdam/conv2d_4/bias/v"Adam/batch_normalization_6/gamma/v!Adam/batch_normalization_6/beta/vAdam/conv2d_5/kernel/vAdam/conv2d_5/bias/v"Adam/batch_normalization_7/gamma/v!Adam/batch_normalization_7/beta/vAdam/dense_3/kernel/vAdam/dense_3/bias/vAdam/dense_4/kernel/vAdam/dense_4/bias/vAdam/dense_5/kernel/vAdam/dense_5/bias/v*Y
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
GPU 2J 8В **
f%R#
!__inference__traced_restore_47366°╩
■
k
I__inference_stft_magnitude_layer_call_and_return_conditional_losses_43739
stft_1_input
identity▌
stft_1/PartitionedCallPartitionedCallstft_1_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ╖Б* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_stft_1_layer_call_and_return_conditional_losses_437112
stft_1/PartitionedCall 
magnitude_1/PartitionedCallPartitionedCallstft_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ╖Б* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_magnitude_1_layer_call_and_return_conditional_losses_437242
magnitude_1/PartitionedCallВ
IdentityIdentity$magnitude_1/PartitionedCall:output:0*
T0*1
_output_shapes
:         ╖Б2

Identity"
identityIdentity:output:0*,
_input_shapes
:         Ат	:[ W
-
_output_shapes
:         Ат	
&
_user_specified_namestft_1_input
А
є
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_44287

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1╠
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:         ╖Б:::::*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3▄
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*1
_output_shapes
:         ╖Б2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:         ╖Б::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:         ╖Б
 
_user_specified_nameinputs
╗U
я	
G__inference_sequential_1_layer_call_and_return_conditional_losses_45080

inputs
batch_normalization_4_45007
batch_normalization_4_45009
batch_normalization_4_45011
batch_normalization_4_45013
conv2d_3_45016
conv2d_3_45018
batch_normalization_5_45022
batch_normalization_5_45024
batch_normalization_5_45026
batch_normalization_5_45028
conv2d_4_45031
conv2d_4_45033
batch_normalization_6_45037
batch_normalization_6_45039
batch_normalization_6_45041
batch_normalization_6_45043
conv2d_5_45046
conv2d_5_45048
batch_normalization_7_45052
batch_normalization_7_45054
batch_normalization_7_45056
batch_normalization_7_45058
dense_3_45062
dense_3_45064
dense_4_45068
dense_4_45070
dense_5_45074
dense_5_45076
identityИв-batch_normalization_4/StatefulPartitionedCallв-batch_normalization_5/StatefulPartitionedCallв-batch_normalization_6/StatefulPartitionedCallв-batch_normalization_7/StatefulPartitionedCallв conv2d_3/StatefulPartitionedCallв conv2d_4/StatefulPartitionedCallв conv2d_5/StatefulPartitionedCallвdense_3/StatefulPartitionedCallвdense_4/StatefulPartitionedCallвdense_5/StatefulPartitionedCall▄
reshape_1/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:         Ат	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_reshape_1_layer_call_and_return_conditional_losses_442312
reshape_1/PartitionedCallЛ
stft_magnitude/PartitionedCallPartitionedCall"reshape_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ╖Б* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_stft_magnitude_layer_call_and_return_conditional_losses_437592 
stft_magnitude/PartitionedCall╗
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall'stft_magnitude/PartitionedCall:output:0batch_normalization_4_45007batch_normalization_4_45009batch_normalization_4_45011batch_normalization_4_45013*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ╖Б*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Y
fTRR
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_442872/
-batch_normalization_4/StatefulPartitionedCall╦
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0conv2d_3_45016conv2d_3_45018*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ╖Б*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_conv2d_3_layer_call_and_return_conditional_losses_443342"
 conv2d_3/StatefulPartitionedCallФ
max_pooling2d_3/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         gл* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_438722!
max_pooling2d_3/PartitionedCall╗
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_3/PartitionedCall:output:0batch_normalization_5_45022batch_normalization_5_45024batch_normalization_5_45026batch_normalization_5_45028*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         gл*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Y
fTRR
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_443882/
-batch_normalization_5/StatefulPartitionedCall╩
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0conv2d_4_45031conv2d_4_45033*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         gл*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_conv2d_4_layer_call_and_return_conditional_losses_444352"
 conv2d_4/StatefulPartitionedCallУ
max_pooling2d_4/PartitionedCallPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         39* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_439882!
max_pooling2d_4/PartitionedCall║
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_4/PartitionedCall:output:0batch_normalization_6_45037batch_normalization_6_45039batch_normalization_6_45041batch_normalization_6_45043*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         39*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Y
fTRR
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_444892/
-batch_normalization_6/StatefulPartitionedCall╔
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:0conv2d_5_45046conv2d_5_45048*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         39 *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_conv2d_5_layer_call_and_return_conditional_losses_445362"
 conv2d_5/StatefulPartitionedCallУ
max_pooling2d_5/PartitionedCallPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_441042!
max_pooling2d_5/PartitionedCall║
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_5/PartitionedCall:output:0batch_normalization_7_45052batch_normalization_7_45054batch_normalization_7_45056batch_normalization_7_45058*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Y
fTRR
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_445902/
-batch_normalization_7/StatefulPartitionedCallЗ
flatten_1/PartitionedCallPartitionedCall6batch_normalization_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         рv* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_446322
flatten_1/PartitionedCallи
dense_3/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_3_45062dense_3_45064*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_446512!
dense_3/StatefulPartitionedCall°
dropout_2/PartitionedCallPartitionedCall(dense_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_446842
dropout_2/PartitionedCallи
dense_4/StatefulPartitionedCallStatefulPartitionedCall"dropout_2/PartitionedCall:output:0dense_4_45068dense_4_45070*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_447082!
dense_4/StatefulPartitionedCall°
dropout_3/PartitionedCallPartitionedCall(dense_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dropout_3_layer_call_and_return_conditional_losses_447412
dropout_3/PartitionedCallи
dense_5/StatefulPartitionedCallStatefulPartitionedCall"dropout_3/PartitionedCall:output:0dense_5_45074dense_5_45076*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dense_5_layer_call_and_return_conditional_losses_447652!
dense_5/StatefulPartitionedCallЛ
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall.^batch_normalization_6/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*Ъ
_input_shapesИ
Е:         Ат	::::::::::::::::::::::::::::2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2^
-batch_normalization_6/StatefulPartitionedCall-batch_normalization_6/StatefulPartitionedCall2^
-batch_normalization_7/StatefulPartitionedCall-batch_normalization_7/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall:Q M
)
_output_shapes
:         Ат	
 
_user_specified_nameinputs
╟
b
D__inference_dropout_3_layer_call_and_return_conditional_losses_44741

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:         2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:         2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
█
и
5__inference_batch_normalization_4_layer_call_fn_46165

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCallв
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ╖Б*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Y
fTRR
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_442692
StatefulPartitionedCallШ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:         ╖Б2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:         ╖Б::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:         ╖Б
 
_user_specified_nameinputs
ч
б
,__inference_sequential_1_layer_call_fn_45139
reshape_1_input
unknown
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

unknown_26
identityИвStatefulPartitionedCallу
StatefulPartitionedCallStatefulPartitionedCallreshape_1_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_26*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *>
_read_only_resource_inputs 
	
*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_450802
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*Ъ
_input_shapesИ
Е:         Ат	::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
)
_output_shapes
:         Ат	
)
_user_specified_namereshape_1_input
Т
E
)__inference_dropout_2_layer_call_fn_46680

inputs
identity┬
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_446842
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:         @2

Identity"
identityIdentity:output:0*&
_input_shapes
:         @:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
Э
и
5__inference_batch_normalization_5_layer_call_fn_46262

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCall┤
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Y
fTRR
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_439712
StatefulPartitionedCallи
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
Ю
b
)__inference_dropout_3_layer_call_fn_46722

inputs
identityИвStatefulPartitionedCall┌
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dropout_3_layer_call_and_return_conditional_losses_447362
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*&
_input_shapes
:         22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
╠
Ч
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_44056

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
FusedBatchNormV3н
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue╗
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1Р
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
Д
Ч
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_46430

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1╪
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         39:::::*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
FusedBatchNormV3н
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue╗
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1■
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:         392

Identity"
identityIdentity:output:0*>
_input_shapes-
+:         39::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:         39
 
_user_specified_nameinputs
А
f
J__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_44104

inputs
identityн
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
2	
MaxPoolЗ
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
╒
и
5__inference_batch_normalization_6_layer_call_fn_46474

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCallв
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         39*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Y
fTRR
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_444892
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:         392

Identity"
identityIdentity:output:0*>
_input_shapes-
+:         39::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         39
 
_user_specified_nameinputs
°
є
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_46596

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1╩
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:          : : : : :*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3┌
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:          2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:          ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:          
 
_user_specified_nameinputs
╪
|
'__inference_dense_5_layer_call_fn_46747

inputs
unknown
	unknown_0
identityИвStatefulPartitionedCallЄ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dense_5_layer_call_and_return_conditional_losses_447652
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*.
_input_shapes
:         ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
рЯ
X
A__inference_stft_1_layer_call_and_return_conditional_losses_43711
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
:         Ат	2
	transposeЛ
"stft_1_tf.signal.stft/frame_lengthConst*
_output_shapes
: *
dtype0*
value
B :А2$
"stft_1_tf.signal.stft/frame_lengthЗ
 stft_1_tf.signal.stft/frame_stepConst*
_output_shapes
: *
dtype0*
value
B :А2"
 stft_1_tf.signal.stft/frame_stepЗ
 stft_1_tf.signal.stft/fft_lengthConst*
_output_shapes
: *
dtype0*
value
B :А2"
 stft_1_tf.signal.stft/fft_lengthП
 stft_1_tf.signal.stft/frame/axisConst*
_output_shapes
: *
dtype0*
valueB :
         2"
 stft_1_tf.signal.stft/frame/axisГ
!stft_1_tf.signal.stft/frame/ShapeShapetranspose:y:0*
T0*
_output_shapes
:2#
!stft_1_tf.signal.stft/frame/ShapeЖ
 stft_1_tf.signal.stft/frame/RankConst*
_output_shapes
: *
dtype0*
value	B :2"
 stft_1_tf.signal.stft/frame/RankФ
'stft_1_tf.signal.stft/frame/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2)
'stft_1_tf.signal.stft/frame/range/startФ
'stft_1_tf.signal.stft/frame/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2)
'stft_1_tf.signal.stft/frame/range/delta·
!stft_1_tf.signal.stft/frame/rangeRange0stft_1_tf.signal.stft/frame/range/start:output:0)stft_1_tf.signal.stft/frame/Rank:output:00stft_1_tf.signal.stft/frame/range/delta:output:0*
_output_shapes
:2#
!stft_1_tf.signal.stft/frame/range╡
/stft_1_tf.signal.stft/frame/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
         21
/stft_1_tf.signal.stft/frame/strided_slice/stack░
1stft_1_tf.signal.stft/frame/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 23
1stft_1_tf.signal.stft/frame/strided_slice/stack_1░
1stft_1_tf.signal.stft/frame/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1stft_1_tf.signal.stft/frame/strided_slice/stack_2К
)stft_1_tf.signal.stft/frame/strided_sliceStridedSlice*stft_1_tf.signal.stft/frame/range:output:08stft_1_tf.signal.stft/frame/strided_slice/stack:output:0:stft_1_tf.signal.stft/frame/strided_slice/stack_1:output:0:stft_1_tf.signal.stft/frame/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)stft_1_tf.signal.stft/frame/strided_sliceИ
!stft_1_tf.signal.stft/frame/sub/yConst*
_output_shapes
: *
dtype0*
value	B :2#
!stft_1_tf.signal.stft/frame/sub/y┴
stft_1_tf.signal.stft/frame/subSub)stft_1_tf.signal.stft/frame/Rank:output:0*stft_1_tf.signal.stft/frame/sub/y:output:0*
T0*
_output_shapes
: 2!
stft_1_tf.signal.stft/frame/sub╟
!stft_1_tf.signal.stft/frame/sub_1Sub#stft_1_tf.signal.stft/frame/sub:z:02stft_1_tf.signal.stft/frame/strided_slice:output:0*
T0*
_output_shapes
: 2#
!stft_1_tf.signal.stft/frame/sub_1О
$stft_1_tf.signal.stft/frame/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2&
$stft_1_tf.signal.stft/frame/packed/1И
"stft_1_tf.signal.stft/frame/packedPack2stft_1_tf.signal.stft/frame/strided_slice:output:0-stft_1_tf.signal.stft/frame/packed/1:output:0%stft_1_tf.signal.stft/frame/sub_1:z:0*
N*
T0*
_output_shapes
:2$
"stft_1_tf.signal.stft/frame/packedЬ
+stft_1_tf.signal.stft/frame/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2-
+stft_1_tf.signal.stft/frame/split/split_dimл
!stft_1_tf.signal.stft/frame/splitSplitV*stft_1_tf.signal.stft/frame/Shape:output:0+stft_1_tf.signal.stft/frame/packed:output:04stft_1_tf.signal.stft/frame/split/split_dim:output:0*
T0*

Tlen0*$
_output_shapes
::: *
	num_split2#
!stft_1_tf.signal.stft/frame/splitЩ
)stft_1_tf.signal.stft/frame/Reshape/shapeConst*
_output_shapes
: *
dtype0*
valueB 2+
)stft_1_tf.signal.stft/frame/Reshape/shapeЭ
+stft_1_tf.signal.stft/frame/Reshape/shape_1Const*
_output_shapes
: *
dtype0*
valueB 2-
+stft_1_tf.signal.stft/frame/Reshape/shape_1╪
#stft_1_tf.signal.stft/frame/ReshapeReshape*stft_1_tf.signal.stft/frame/split:output:14stft_1_tf.signal.stft/frame/Reshape/shape_1:output:0*
T0*
_output_shapes
: 2%
#stft_1_tf.signal.stft/frame/ReshapeЖ
 stft_1_tf.signal.stft/frame/SizeConst*
_output_shapes
: *
dtype0*
value	B :2"
 stft_1_tf.signal.stft/frame/SizeК
"stft_1_tf.signal.stft/frame/Size_1Const*
_output_shapes
: *
dtype0*
value	B : 2$
"stft_1_tf.signal.stft/frame/Size_1╔
!stft_1_tf.signal.stft/frame/sub_2Sub,stft_1_tf.signal.stft/frame/Reshape:output:0+stft_1_tf.signal.stft/frame_length:output:0*
T0*
_output_shapes
: 2#
!stft_1_tf.signal.stft/frame/sub_2╦
$stft_1_tf.signal.stft/frame/floordivFloorDiv%stft_1_tf.signal.stft/frame/sub_2:z:0)stft_1_tf.signal.stft/frame_step:output:0*
T0*
_output_shapes
: 2&
$stft_1_tf.signal.stft/frame/floordivИ
!stft_1_tf.signal.stft/frame/add/xConst*
_output_shapes
: *
dtype0*
value	B :2#
!stft_1_tf.signal.stft/frame/add/x┬
stft_1_tf.signal.stft/frame/addAddV2*stft_1_tf.signal.stft/frame/add/x:output:0(stft_1_tf.signal.stft/frame/floordiv:z:0*
T0*
_output_shapes
: 2!
stft_1_tf.signal.stft/frame/addР
%stft_1_tf.signal.stft/frame/Maximum/xConst*
_output_shapes
: *
dtype0*
value	B : 2'
%stft_1_tf.signal.stft/frame/Maximum/x╦
#stft_1_tf.signal.stft/frame/MaximumMaximum.stft_1_tf.signal.stft/frame/Maximum/x:output:0#stft_1_tf.signal.stft/frame/add:z:0*
T0*
_output_shapes
: 2%
#stft_1_tf.signal.stft/frame/MaximumС
%stft_1_tf.signal.stft/frame/gcd/ConstConst*
_output_shapes
: *
dtype0*
value
B :А2'
%stft_1_tf.signal.stft/frame/gcd/ConstЧ
(stft_1_tf.signal.stft/frame/floordiv_1/yConst*
_output_shapes
: *
dtype0*
value
B :А2*
(stft_1_tf.signal.stft/frame/floordiv_1/y▌
&stft_1_tf.signal.stft/frame/floordiv_1FloorDiv+stft_1_tf.signal.stft/frame_length:output:01stft_1_tf.signal.stft/frame/floordiv_1/y:output:0*
T0*
_output_shapes
: 2(
&stft_1_tf.signal.stft/frame/floordiv_1Ч
(stft_1_tf.signal.stft/frame/floordiv_2/yConst*
_output_shapes
: *
dtype0*
value
B :А2*
(stft_1_tf.signal.stft/frame/floordiv_2/y█
&stft_1_tf.signal.stft/frame/floordiv_2FloorDiv)stft_1_tf.signal.stft/frame_step:output:01stft_1_tf.signal.stft/frame/floordiv_2/y:output:0*
T0*
_output_shapes
: 2(
&stft_1_tf.signal.stft/frame/floordiv_2Ч
(stft_1_tf.signal.stft/frame/floordiv_3/yConst*
_output_shapes
: *
dtype0*
value
B :А2*
(stft_1_tf.signal.stft/frame/floordiv_3/y▐
&stft_1_tf.signal.stft/frame/floordiv_3FloorDiv,stft_1_tf.signal.stft/frame/Reshape:output:01stft_1_tf.signal.stft/frame/floordiv_3/y:output:0*
T0*
_output_shapes
: 2(
&stft_1_tf.signal.stft/frame/floordiv_3Й
!stft_1_tf.signal.stft/frame/mul/yConst*
_output_shapes
: *
dtype0*
value
B :А2#
!stft_1_tf.signal.stft/frame/mul/y┬
stft_1_tf.signal.stft/frame/mulMul*stft_1_tf.signal.stft/frame/floordiv_3:z:0*stft_1_tf.signal.stft/frame/mul/y:output:0*
T0*
_output_shapes
: 2!
stft_1_tf.signal.stft/frame/mul╡
+stft_1_tf.signal.stft/frame/concat/values_1Pack#stft_1_tf.signal.stft/frame/mul:z:0*
N*
T0*
_output_shapes
:2-
+stft_1_tf.signal.stft/frame/concat/values_1Ф
'stft_1_tf.signal.stft/frame/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2)
'stft_1_tf.signal.stft/frame/concat/axis┬
"stft_1_tf.signal.stft/frame/concatConcatV2*stft_1_tf.signal.stft/frame/split:output:04stft_1_tf.signal.stft/frame/concat/values_1:output:0*stft_1_tf.signal.stft/frame/split:output:20stft_1_tf.signal.stft/frame/concat/axis:output:0*
N*
T0*
_output_shapes
:2$
"stft_1_tf.signal.stft/frame/concatе
/stft_1_tf.signal.stft/frame/concat_1/values_1/1Const*
_output_shapes
: *
dtype0*
value
B :А21
/stft_1_tf.signal.stft/frame/concat_1/values_1/1·
-stft_1_tf.signal.stft/frame/concat_1/values_1Pack*stft_1_tf.signal.stft/frame/floordiv_3:z:08stft_1_tf.signal.stft/frame/concat_1/values_1/1:output:0*
N*
T0*
_output_shapes
:2/
-stft_1_tf.signal.stft/frame/concat_1/values_1Ш
)stft_1_tf.signal.stft/frame/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2+
)stft_1_tf.signal.stft/frame/concat_1/axis╩
$stft_1_tf.signal.stft/frame/concat_1ConcatV2*stft_1_tf.signal.stft/frame/split:output:06stft_1_tf.signal.stft/frame/concat_1/values_1:output:0*stft_1_tf.signal.stft/frame/split:output:22stft_1_tf.signal.stft/frame/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2&
$stft_1_tf.signal.stft/frame/concat_1Ъ
&stft_1_tf.signal.stft/frame/zeros_likeConst*
_output_shapes
:*
dtype0*
valueB: 2(
&stft_1_tf.signal.stft/frame/zeros_likeд
+stft_1_tf.signal.stft/frame/ones_like/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2-
+stft_1_tf.signal.stft/frame/ones_like/ShapeЬ
+stft_1_tf.signal.stft/frame/ones_like/ConstConst*
_output_shapes
: *
dtype0*
value	B :2-
+stft_1_tf.signal.stft/frame/ones_like/Constч
%stft_1_tf.signal.stft/frame/ones_likeFill4stft_1_tf.signal.stft/frame/ones_like/Shape:output:04stft_1_tf.signal.stft/frame/ones_like/Const:output:0*
T0*
_output_shapes
:2'
%stft_1_tf.signal.stft/frame/ones_like╓
(stft_1_tf.signal.stft/frame/StridedSliceStridedSlicetranspose:y:0/stft_1_tf.signal.stft/frame/zeros_like:output:0+stft_1_tf.signal.stft/frame/concat:output:0.stft_1_tf.signal.stft/frame/ones_like:output:0*
Index0*
T0*=
_output_shapes+
):'                           2*
(stft_1_tf.signal.stft/frame/StridedSliceИ
%stft_1_tf.signal.stft/frame/Reshape_1Reshape1stft_1_tf.signal.stft/frame/StridedSlice:output:0-stft_1_tf.signal.stft/frame/concat_1:output:0*
T0*B
_output_shapes0
.:,                           А2'
%stft_1_tf.signal.stft/frame/Reshape_1Ш
)stft_1_tf.signal.stft/frame/range_1/startConst*
_output_shapes
: *
dtype0*
value	B : 2+
)stft_1_tf.signal.stft/frame/range_1/startШ
)stft_1_tf.signal.stft/frame/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :2+
)stft_1_tf.signal.stft/frame/range_1/deltaЙ
#stft_1_tf.signal.stft/frame/range_1Range2stft_1_tf.signal.stft/frame/range_1/start:output:0'stft_1_tf.signal.stft/frame/Maximum:z:02stft_1_tf.signal.stft/frame/range_1/delta:output:0*#
_output_shapes
:         2%
#stft_1_tf.signal.stft/frame/range_1╒
!stft_1_tf.signal.stft/frame/mul_1Mul,stft_1_tf.signal.stft/frame/range_1:output:0*stft_1_tf.signal.stft/frame/floordiv_2:z:0*
T0*#
_output_shapes
:         2#
!stft_1_tf.signal.stft/frame/mul_1а
-stft_1_tf.signal.stft/frame/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2/
-stft_1_tf.signal.stft/frame/Reshape_2/shape/1ё
+stft_1_tf.signal.stft/frame/Reshape_2/shapePack'stft_1_tf.signal.stft/frame/Maximum:z:06stft_1_tf.signal.stft/frame/Reshape_2/shape/1:output:0*
N*
T0*
_output_shapes
:2-
+stft_1_tf.signal.stft/frame/Reshape_2/shapeш
%stft_1_tf.signal.stft/frame/Reshape_2Reshape%stft_1_tf.signal.stft/frame/mul_1:z:04stft_1_tf.signal.stft/frame/Reshape_2/shape:output:0*
T0*'
_output_shapes
:         2'
%stft_1_tf.signal.stft/frame/Reshape_2Ш
)stft_1_tf.signal.stft/frame/range_2/startConst*
_output_shapes
: *
dtype0*
value	B : 2+
)stft_1_tf.signal.stft/frame/range_2/startШ
)stft_1_tf.signal.stft/frame/range_2/deltaConst*
_output_shapes
: *
dtype0*
value	B :2+
)stft_1_tf.signal.stft/frame/range_2/deltaГ
#stft_1_tf.signal.stft/frame/range_2Range2stft_1_tf.signal.stft/frame/range_2/start:output:0*stft_1_tf.signal.stft/frame/floordiv_1:z:02stft_1_tf.signal.stft/frame/range_2/delta:output:0*
_output_shapes
:2%
#stft_1_tf.signal.stft/frame/range_2а
-stft_1_tf.signal.stft/frame/Reshape_3/shape/0Const*
_output_shapes
: *
dtype0*
value	B :2/
-stft_1_tf.signal.stft/frame/Reshape_3/shape/0Ї
+stft_1_tf.signal.stft/frame/Reshape_3/shapePack6stft_1_tf.signal.stft/frame/Reshape_3/shape/0:output:0*stft_1_tf.signal.stft/frame/floordiv_1:z:0*
N*
T0*
_output_shapes
:2-
+stft_1_tf.signal.stft/frame/Reshape_3/shapeц
%stft_1_tf.signal.stft/frame/Reshape_3Reshape,stft_1_tf.signal.stft/frame/range_2:output:04stft_1_tf.signal.stft/frame/Reshape_3/shape:output:0*
T0*
_output_shapes

:2'
%stft_1_tf.signal.stft/frame/Reshape_3с
!stft_1_tf.signal.stft/frame/add_1AddV2.stft_1_tf.signal.stft/frame/Reshape_2:output:0.stft_1_tf.signal.stft/frame/Reshape_3:output:0*
T0*'
_output_shapes
:         2#
!stft_1_tf.signal.stft/frame/add_1╫
$stft_1_tf.signal.stft/frame/GatherV2GatherV2.stft_1_tf.signal.stft/frame/Reshape_1:output:0%stft_1_tf.signal.stft/frame/add_1:z:02stft_1_tf.signal.stft/frame/strided_slice:output:0*
Taxis0*
Tindices0*
Tparams0*F
_output_shapes4
2:0                           А2&
$stft_1_tf.signal.stft/frame/GatherV2ъ
-stft_1_tf.signal.stft/frame/concat_2/values_1Pack'stft_1_tf.signal.stft/frame/Maximum:z:0+stft_1_tf.signal.stft/frame_length:output:0*
N*
T0*
_output_shapes
:2/
-stft_1_tf.signal.stft/frame/concat_2/values_1Ш
)stft_1_tf.signal.stft/frame/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2+
)stft_1_tf.signal.stft/frame/concat_2/axis╩
$stft_1_tf.signal.stft/frame/concat_2ConcatV2*stft_1_tf.signal.stft/frame/split:output:06stft_1_tf.signal.stft/frame/concat_2/values_1:output:0*stft_1_tf.signal.stft/frame/split:output:22stft_1_tf.signal.stft/frame/concat_2/axis:output:0*
N*
T0*
_output_shapes
:2&
$stft_1_tf.signal.stft/frame/concat_2є
%stft_1_tf.signal.stft/frame/Reshape_4Reshape-stft_1_tf.signal.stft/frame/GatherV2:output:0-stft_1_tf.signal.stft/frame/concat_2:output:0*
T0*1
_output_shapes
:         ╖А2'
%stft_1_tf.signal.stft/frame/Reshape_4Ъ
*stft_1_tf.signal.stft/hann_window/periodicConst*
_output_shapes
: *
dtype0
*
value	B
 Z2,
*stft_1_tf.signal.stft/hann_window/periodic╜
&stft_1_tf.signal.stft/hann_window/CastCast3stft_1_tf.signal.stft/hann_window/periodic:output:0*

DstT0*

SrcT0
*
_output_shapes
: 2(
&stft_1_tf.signal.stft/hann_window/CastЮ
,stft_1_tf.signal.stft/hann_window/FloorMod/yConst*
_output_shapes
: *
dtype0*
value	B :2.
,stft_1_tf.signal.stft/hann_window/FloorMod/yщ
*stft_1_tf.signal.stft/hann_window/FloorModFloorMod+stft_1_tf.signal.stft/frame_length:output:05stft_1_tf.signal.stft/hann_window/FloorMod/y:output:0*
T0*
_output_shapes
: 2,
*stft_1_tf.signal.stft/hann_window/FloorModФ
'stft_1_tf.signal.stft/hann_window/sub/xConst*
_output_shapes
: *
dtype0*
value	B :2)
'stft_1_tf.signal.stft/hann_window/sub/x╪
%stft_1_tf.signal.stft/hann_window/subSub0stft_1_tf.signal.stft/hann_window/sub/x:output:0.stft_1_tf.signal.stft/hann_window/FloorMod:z:0*
T0*
_output_shapes
: 2'
%stft_1_tf.signal.stft/hann_window/sub═
%stft_1_tf.signal.stft/hann_window/mulMul*stft_1_tf.signal.stft/hann_window/Cast:y:0)stft_1_tf.signal.stft/hann_window/sub:z:0*
T0*
_output_shapes
: 2'
%stft_1_tf.signal.stft/hann_window/mul╨
%stft_1_tf.signal.stft/hann_window/addAddV2+stft_1_tf.signal.stft/frame_length:output:0)stft_1_tf.signal.stft/hann_window/mul:z:0*
T0*
_output_shapes
: 2'
%stft_1_tf.signal.stft/hann_window/addШ
)stft_1_tf.signal.stft/hann_window/sub_1/yConst*
_output_shapes
: *
dtype0*
value	B :2+
)stft_1_tf.signal.stft/hann_window/sub_1/y┘
'stft_1_tf.signal.stft/hann_window/sub_1Sub)stft_1_tf.signal.stft/hann_window/add:z:02stft_1_tf.signal.stft/hann_window/sub_1/y:output:0*
T0*
_output_shapes
: 2)
'stft_1_tf.signal.stft/hann_window/sub_1╣
(stft_1_tf.signal.stft/hann_window/Cast_1Cast+stft_1_tf.signal.stft/hann_window/sub_1:z:0*

DstT0*

SrcT0*
_output_shapes
: 2*
(stft_1_tf.signal.stft/hann_window/Cast_1а
-stft_1_tf.signal.stft/hann_window/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2/
-stft_1_tf.signal.stft/hann_window/range/startа
-stft_1_tf.signal.stft/hann_window/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2/
-stft_1_tf.signal.stft/hann_window/range/deltaХ
'stft_1_tf.signal.stft/hann_window/rangeRange6stft_1_tf.signal.stft/hann_window/range/start:output:0+stft_1_tf.signal.stft/frame_length:output:06stft_1_tf.signal.stft/hann_window/range/delta:output:0*
_output_shapes	
:А2)
'stft_1_tf.signal.stft/hann_window/range├
(stft_1_tf.signal.stft/hann_window/Cast_2Cast0stft_1_tf.signal.stft/hann_window/range:output:0*

DstT0*

SrcT0*
_output_shapes	
:А2*
(stft_1_tf.signal.stft/hann_window/Cast_2Ч
'stft_1_tf.signal.stft/hann_window/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *█╔@2)
'stft_1_tf.signal.stft/hann_window/Const▀
'stft_1_tf.signal.stft/hann_window/mul_1Mul0stft_1_tf.signal.stft/hann_window/Const:output:0,stft_1_tf.signal.stft/hann_window/Cast_2:y:0*
T0*
_output_shapes	
:А2)
'stft_1_tf.signal.stft/hann_window/mul_1т
)stft_1_tf.signal.stft/hann_window/truedivRealDiv+stft_1_tf.signal.stft/hann_window/mul_1:z:0,stft_1_tf.signal.stft/hann_window/Cast_1:y:0*
T0*
_output_shapes	
:А2+
)stft_1_tf.signal.stft/hann_window/truedivк
%stft_1_tf.signal.stft/hann_window/CosCos-stft_1_tf.signal.stft/hann_window/truediv:z:0*
T0*
_output_shapes	
:А2'
%stft_1_tf.signal.stft/hann_window/CosЫ
)stft_1_tf.signal.stft/hann_window/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2+
)stft_1_tf.signal.stft/hann_window/mul_2/x▐
'stft_1_tf.signal.stft/hann_window/mul_2Mul2stft_1_tf.signal.stft/hann_window/mul_2/x:output:0)stft_1_tf.signal.stft/hann_window/Cos:y:0*
T0*
_output_shapes	
:А2)
'stft_1_tf.signal.stft/hann_window/mul_2Ы
)stft_1_tf.signal.stft/hann_window/sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2+
)stft_1_tf.signal.stft/hann_window/sub_2/xр
'stft_1_tf.signal.stft/hann_window/sub_2Sub2stft_1_tf.signal.stft/hann_window/sub_2/x:output:0+stft_1_tf.signal.stft/hann_window/mul_2:z:0*
T0*
_output_shapes	
:А2)
'stft_1_tf.signal.stft/hann_window/sub_2╓
stft_1_tf.signal.stft/mulMul.stft_1_tf.signal.stft/frame/Reshape_4:output:0+stft_1_tf.signal.stft/hann_window/sub_2:z:0*
T0*1
_output_shapes
:         ╖А2
stft_1_tf.signal.stft/mulз
!stft_1_tf.signal.stft/rfft/packedPack)stft_1_tf.signal.stft/fft_length:output:0*
N*
T0*
_output_shapes
:2#
!stft_1_tf.signal.stft/rfft/packedЩ
%stft_1_tf.signal.stft/rfft/fft_lengthConst*
_output_shapes
:*
dtype0*
valueB:А2'
%stft_1_tf.signal.stft/rfft/fft_length┬
stft_1_tf.signal.stft/rfftRFFTstft_1_tf.signal.stft/mul:z:0.stft_1_tf.signal.stft/rfft/fft_length:output:0*1
_output_shapes
:         ╖Б2
stft_1_tf.signal.stft/rfft}
transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             2
transpose_1/permг
transpose_1	Transpose#stft_1_tf.signal.stft/rfft:output:0transpose_1/perm:output:0*
T0*1
_output_shapes
:         ╖Б2
transpose_1m
IdentityIdentitytranspose_1:y:0*
T0*1
_output_shapes
:         ╖Б2

Identity"
identityIdentity:output:0*,
_input_shapes
:         Ат	:P L
-
_output_shapes
:         Ат	

_user_specified_namex
╠
Ч
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_43940

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
FusedBatchNormV3н
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue╗
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1Р
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
Э
=
&__inference_stft_1_layer_call_fn_46861
x
identity─
PartitionedCallPartitionedCallx*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ╖Б* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_stft_1_layer_call_and_return_conditional_losses_437112
PartitionedCallv
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:         ╖Б2

Identity"
identityIdentity:output:0*,
_input_shapes
:         Ат	:P L
-
_output_shapes
:         Ат	

_user_specified_namex
д╟
Е+
!__inference__traced_restore_47366
file_prefix0
,assignvariableop_batch_normalization_4_gamma1
-assignvariableop_1_batch_normalization_4_beta8
4assignvariableop_2_batch_normalization_4_moving_mean<
8assignvariableop_3_batch_normalization_4_moving_variance&
"assignvariableop_4_conv2d_3_kernel$
 assignvariableop_5_conv2d_3_bias2
.assignvariableop_6_batch_normalization_5_gamma1
-assignvariableop_7_batch_normalization_5_beta8
4assignvariableop_8_batch_normalization_5_moving_mean<
8assignvariableop_9_batch_normalization_5_moving_variance'
#assignvariableop_10_conv2d_4_kernel%
!assignvariableop_11_conv2d_4_bias3
/assignvariableop_12_batch_normalization_6_gamma2
.assignvariableop_13_batch_normalization_6_beta9
5assignvariableop_14_batch_normalization_6_moving_mean=
9assignvariableop_15_batch_normalization_6_moving_variance'
#assignvariableop_16_conv2d_5_kernel%
!assignvariableop_17_conv2d_5_bias3
/assignvariableop_18_batch_normalization_7_gamma2
.assignvariableop_19_batch_normalization_7_beta9
5assignvariableop_20_batch_normalization_7_moving_mean=
9assignvariableop_21_batch_normalization_7_moving_variance&
"assignvariableop_22_dense_3_kernel$
 assignvariableop_23_dense_3_bias&
"assignvariableop_24_dense_4_kernel$
 assignvariableop_25_dense_4_bias&
"assignvariableop_26_dense_5_kernel$
 assignvariableop_27_dense_5_bias!
assignvariableop_28_adam_iter#
assignvariableop_29_adam_beta_1#
assignvariableop_30_adam_beta_2"
assignvariableop_31_adam_decay*
&assignvariableop_32_adam_learning_rate
assignvariableop_33_total
assignvariableop_34_count
assignvariableop_35_total_1
assignvariableop_36_count_1:
6assignvariableop_37_adam_batch_normalization_4_gamma_m9
5assignvariableop_38_adam_batch_normalization_4_beta_m.
*assignvariableop_39_adam_conv2d_3_kernel_m,
(assignvariableop_40_adam_conv2d_3_bias_m:
6assignvariableop_41_adam_batch_normalization_5_gamma_m9
5assignvariableop_42_adam_batch_normalization_5_beta_m.
*assignvariableop_43_adam_conv2d_4_kernel_m,
(assignvariableop_44_adam_conv2d_4_bias_m:
6assignvariableop_45_adam_batch_normalization_6_gamma_m9
5assignvariableop_46_adam_batch_normalization_6_beta_m.
*assignvariableop_47_adam_conv2d_5_kernel_m,
(assignvariableop_48_adam_conv2d_5_bias_m:
6assignvariableop_49_adam_batch_normalization_7_gamma_m9
5assignvariableop_50_adam_batch_normalization_7_beta_m-
)assignvariableop_51_adam_dense_3_kernel_m+
'assignvariableop_52_adam_dense_3_bias_m-
)assignvariableop_53_adam_dense_4_kernel_m+
'assignvariableop_54_adam_dense_4_bias_m-
)assignvariableop_55_adam_dense_5_kernel_m+
'assignvariableop_56_adam_dense_5_bias_m:
6assignvariableop_57_adam_batch_normalization_4_gamma_v9
5assignvariableop_58_adam_batch_normalization_4_beta_v.
*assignvariableop_59_adam_conv2d_3_kernel_v,
(assignvariableop_60_adam_conv2d_3_bias_v:
6assignvariableop_61_adam_batch_normalization_5_gamma_v9
5assignvariableop_62_adam_batch_normalization_5_beta_v.
*assignvariableop_63_adam_conv2d_4_kernel_v,
(assignvariableop_64_adam_conv2d_4_bias_v:
6assignvariableop_65_adam_batch_normalization_6_gamma_v9
5assignvariableop_66_adam_batch_normalization_6_beta_v.
*assignvariableop_67_adam_conv2d_5_kernel_v,
(assignvariableop_68_adam_conv2d_5_bias_v:
6assignvariableop_69_adam_batch_normalization_7_gamma_v9
5assignvariableop_70_adam_batch_normalization_7_beta_v-
)assignvariableop_71_adam_dense_3_kernel_v+
'assignvariableop_72_adam_dense_3_bias_v-
)assignvariableop_73_adam_dense_4_kernel_v+
'assignvariableop_74_adam_dense_4_bias_v-
)assignvariableop_75_adam_dense_5_kernel_v+
'assignvariableop_76_adam_dense_5_bias_v
identity_78ИвAssignVariableOpвAssignVariableOp_1вAssignVariableOp_10вAssignVariableOp_11вAssignVariableOp_12вAssignVariableOp_13вAssignVariableOp_14вAssignVariableOp_15вAssignVariableOp_16вAssignVariableOp_17вAssignVariableOp_18вAssignVariableOp_19вAssignVariableOp_2вAssignVariableOp_20вAssignVariableOp_21вAssignVariableOp_22вAssignVariableOp_23вAssignVariableOp_24вAssignVariableOp_25вAssignVariableOp_26вAssignVariableOp_27вAssignVariableOp_28вAssignVariableOp_29вAssignVariableOp_3вAssignVariableOp_30вAssignVariableOp_31вAssignVariableOp_32вAssignVariableOp_33вAssignVariableOp_34вAssignVariableOp_35вAssignVariableOp_36вAssignVariableOp_37вAssignVariableOp_38вAssignVariableOp_39вAssignVariableOp_4вAssignVariableOp_40вAssignVariableOp_41вAssignVariableOp_42вAssignVariableOp_43вAssignVariableOp_44вAssignVariableOp_45вAssignVariableOp_46вAssignVariableOp_47вAssignVariableOp_48вAssignVariableOp_49вAssignVariableOp_5вAssignVariableOp_50вAssignVariableOp_51вAssignVariableOp_52вAssignVariableOp_53вAssignVariableOp_54вAssignVariableOp_55вAssignVariableOp_56вAssignVariableOp_57вAssignVariableOp_58вAssignVariableOp_59вAssignVariableOp_6вAssignVariableOp_60вAssignVariableOp_61вAssignVariableOp_62вAssignVariableOp_63вAssignVariableOp_64вAssignVariableOp_65вAssignVariableOp_66вAssignVariableOp_67вAssignVariableOp_68вAssignVariableOp_69вAssignVariableOp_7вAssignVariableOp_70вAssignVariableOp_71вAssignVariableOp_72вAssignVariableOp_73вAssignVariableOp_74вAssignVariableOp_75вAssignVariableOp_76вAssignVariableOp_8вAssignVariableOp_9Ф+
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:N*
dtype0*а*
valueЦ*BУ*NB5layer_with_weights-0/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-0/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-0/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesн
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:N*
dtype0*▒
valueзBдNB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices┤
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*╬
_output_shapes╗
╕::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*\
dtypesR
P2N	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identityл
AssignVariableOpAssignVariableOp,assignvariableop_batch_normalization_4_gammaIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1▓
AssignVariableOp_1AssignVariableOp-assignvariableop_1_batch_normalization_4_betaIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2╣
AssignVariableOp_2AssignVariableOp4assignvariableop_2_batch_normalization_4_moving_meanIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3╜
AssignVariableOp_3AssignVariableOp8assignvariableop_3_batch_normalization_4_moving_varianceIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4з
AssignVariableOp_4AssignVariableOp"assignvariableop_4_conv2d_3_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5е
AssignVariableOp_5AssignVariableOp assignvariableop_5_conv2d_3_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6│
AssignVariableOp_6AssignVariableOp.assignvariableop_6_batch_normalization_5_gammaIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7▓
AssignVariableOp_7AssignVariableOp-assignvariableop_7_batch_normalization_5_betaIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8╣
AssignVariableOp_8AssignVariableOp4assignvariableop_8_batch_normalization_5_moving_meanIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9╜
AssignVariableOp_9AssignVariableOp8assignvariableop_9_batch_normalization_5_moving_varianceIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10л
AssignVariableOp_10AssignVariableOp#assignvariableop_10_conv2d_4_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11й
AssignVariableOp_11AssignVariableOp!assignvariableop_11_conv2d_4_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12╖
AssignVariableOp_12AssignVariableOp/assignvariableop_12_batch_normalization_6_gammaIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13╢
AssignVariableOp_13AssignVariableOp.assignvariableop_13_batch_normalization_6_betaIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14╜
AssignVariableOp_14AssignVariableOp5assignvariableop_14_batch_normalization_6_moving_meanIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15┴
AssignVariableOp_15AssignVariableOp9assignvariableop_15_batch_normalization_6_moving_varianceIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16л
AssignVariableOp_16AssignVariableOp#assignvariableop_16_conv2d_5_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17й
AssignVariableOp_17AssignVariableOp!assignvariableop_17_conv2d_5_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18╖
AssignVariableOp_18AssignVariableOp/assignvariableop_18_batch_normalization_7_gammaIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19╢
AssignVariableOp_19AssignVariableOp.assignvariableop_19_batch_normalization_7_betaIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20╜
AssignVariableOp_20AssignVariableOp5assignvariableop_20_batch_normalization_7_moving_meanIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21┴
AssignVariableOp_21AssignVariableOp9assignvariableop_21_batch_normalization_7_moving_varianceIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22к
AssignVariableOp_22AssignVariableOp"assignvariableop_22_dense_3_kernelIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23и
AssignVariableOp_23AssignVariableOp assignvariableop_23_dense_3_biasIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24к
AssignVariableOp_24AssignVariableOp"assignvariableop_24_dense_4_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25и
AssignVariableOp_25AssignVariableOp assignvariableop_25_dense_4_biasIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26к
AssignVariableOp_26AssignVariableOp"assignvariableop_26_dense_5_kernelIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27и
AssignVariableOp_27AssignVariableOp assignvariableop_27_dense_5_biasIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_28е
AssignVariableOp_28AssignVariableOpassignvariableop_28_adam_iterIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29з
AssignVariableOp_29AssignVariableOpassignvariableop_29_adam_beta_1Identity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30з
AssignVariableOp_30AssignVariableOpassignvariableop_30_adam_beta_2Identity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31ж
AssignVariableOp_31AssignVariableOpassignvariableop_31_adam_decayIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32о
AssignVariableOp_32AssignVariableOp&assignvariableop_32_adam_learning_rateIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33б
AssignVariableOp_33AssignVariableOpassignvariableop_33_totalIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34б
AssignVariableOp_34AssignVariableOpassignvariableop_34_countIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35г
AssignVariableOp_35AssignVariableOpassignvariableop_35_total_1Identity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36г
AssignVariableOp_36AssignVariableOpassignvariableop_36_count_1Identity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37╛
AssignVariableOp_37AssignVariableOp6assignvariableop_37_adam_batch_normalization_4_gamma_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38╜
AssignVariableOp_38AssignVariableOp5assignvariableop_38_adam_batch_normalization_4_beta_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39▓
AssignVariableOp_39AssignVariableOp*assignvariableop_39_adam_conv2d_3_kernel_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40░
AssignVariableOp_40AssignVariableOp(assignvariableop_40_adam_conv2d_3_bias_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41╛
AssignVariableOp_41AssignVariableOp6assignvariableop_41_adam_batch_normalization_5_gamma_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42╜
AssignVariableOp_42AssignVariableOp5assignvariableop_42_adam_batch_normalization_5_beta_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43▓
AssignVariableOp_43AssignVariableOp*assignvariableop_43_adam_conv2d_4_kernel_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44░
AssignVariableOp_44AssignVariableOp(assignvariableop_44_adam_conv2d_4_bias_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45╛
AssignVariableOp_45AssignVariableOp6assignvariableop_45_adam_batch_normalization_6_gamma_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46╜
AssignVariableOp_46AssignVariableOp5assignvariableop_46_adam_batch_normalization_6_beta_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47▓
AssignVariableOp_47AssignVariableOp*assignvariableop_47_adam_conv2d_5_kernel_mIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48░
AssignVariableOp_48AssignVariableOp(assignvariableop_48_adam_conv2d_5_bias_mIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49╛
AssignVariableOp_49AssignVariableOp6assignvariableop_49_adam_batch_normalization_7_gamma_mIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50╜
AssignVariableOp_50AssignVariableOp5assignvariableop_50_adam_batch_normalization_7_beta_mIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51▒
AssignVariableOp_51AssignVariableOp)assignvariableop_51_adam_dense_3_kernel_mIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52п
AssignVariableOp_52AssignVariableOp'assignvariableop_52_adam_dense_3_bias_mIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53▒
AssignVariableOp_53AssignVariableOp)assignvariableop_53_adam_dense_4_kernel_mIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54п
AssignVariableOp_54AssignVariableOp'assignvariableop_54_adam_dense_4_bias_mIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55▒
AssignVariableOp_55AssignVariableOp)assignvariableop_55_adam_dense_5_kernel_mIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56п
AssignVariableOp_56AssignVariableOp'assignvariableop_56_adam_dense_5_bias_mIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57╛
AssignVariableOp_57AssignVariableOp6assignvariableop_57_adam_batch_normalization_4_gamma_vIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58╜
AssignVariableOp_58AssignVariableOp5assignvariableop_58_adam_batch_normalization_4_beta_vIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59▓
AssignVariableOp_59AssignVariableOp*assignvariableop_59_adam_conv2d_3_kernel_vIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60░
AssignVariableOp_60AssignVariableOp(assignvariableop_60_adam_conv2d_3_bias_vIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61╛
AssignVariableOp_61AssignVariableOp6assignvariableop_61_adam_batch_normalization_5_gamma_vIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62╜
AssignVariableOp_62AssignVariableOp5assignvariableop_62_adam_batch_normalization_5_beta_vIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63▓
AssignVariableOp_63AssignVariableOp*assignvariableop_63_adam_conv2d_4_kernel_vIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64░
AssignVariableOp_64AssignVariableOp(assignvariableop_64_adam_conv2d_4_bias_vIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_64n
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:2
Identity_65╛
AssignVariableOp_65AssignVariableOp6assignvariableop_65_adam_batch_normalization_6_gamma_vIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_65n
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:2
Identity_66╜
AssignVariableOp_66AssignVariableOp5assignvariableop_66_adam_batch_normalization_6_beta_vIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_66n
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:2
Identity_67▓
AssignVariableOp_67AssignVariableOp*assignvariableop_67_adam_conv2d_5_kernel_vIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_67n
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:2
Identity_68░
AssignVariableOp_68AssignVariableOp(assignvariableop_68_adam_conv2d_5_bias_vIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_68n
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:2
Identity_69╛
AssignVariableOp_69AssignVariableOp6assignvariableop_69_adam_batch_normalization_7_gamma_vIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_69n
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:2
Identity_70╜
AssignVariableOp_70AssignVariableOp5assignvariableop_70_adam_batch_normalization_7_beta_vIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_70n
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:2
Identity_71▒
AssignVariableOp_71AssignVariableOp)assignvariableop_71_adam_dense_3_kernel_vIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_71n
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:2
Identity_72п
AssignVariableOp_72AssignVariableOp'assignvariableop_72_adam_dense_3_bias_vIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_72n
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:2
Identity_73▒
AssignVariableOp_73AssignVariableOp)assignvariableop_73_adam_dense_4_kernel_vIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_73n
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:2
Identity_74п
AssignVariableOp_74AssignVariableOp'assignvariableop_74_adam_dense_4_bias_vIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_74n
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:2
Identity_75▒
AssignVariableOp_75AssignVariableOp)assignvariableop_75_adam_dense_5_kernel_vIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_75n
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:2
Identity_76п
AssignVariableOp_76AssignVariableOp'assignvariableop_76_adam_dense_5_bias_vIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_769
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp№
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
identity_78Identity_78:output:0*╦
_input_shapes╣
╢: :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
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
Д
Ч
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_44471

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1╪
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         39:::::*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
FusedBatchNormV3н
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue╗
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1■
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:         392

Identity"
identityIdentity:output:0*>
_input_shapes-
+:         39::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:         39
 
_user_specified_nameinputs
╝
`
D__inference_flatten_1_layer_call_and_return_conditional_losses_44632

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"    `;  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:         рv2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:         рv2

Identity"
identityIdentity:output:0*.
_input_shapes
:          :W S
/
_output_shapes
:          
 
_user_specified_nameinputs
╙
и
5__inference_batch_normalization_6_layer_call_fn_46461

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCallа
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         39*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Y
fTRR
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_444712
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:         392

Identity"
identityIdentity:output:0*>
_input_shapes-
+:         39::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         39
 
_user_specified_nameinputs
Ы
и
5__inference_batch_normalization_5_layer_call_fn_46249

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCall▓
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Y
fTRR
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_439402
StatefulPartitionedCallи
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
╖
Ш
#__inference_signature_wrapper_45210
reshape_1_input
unknown
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

unknown_26
identityИвStatefulPartitionedCall╝
StatefulPartitionedCallStatefulPartitionedCallreshape_1_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_26*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *>
_read_only_resource_inputs 
	
*-
config_proto

CPU

GPU 2J 8В *)
f$R"
 __inference__wrapped_model_435982
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*Ъ
_input_shapesИ
Е:         Ат	::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
)
_output_shapes
:         Ат	
)
_user_specified_namereshape_1_input
рЯ
X
A__inference_stft_1_layer_call_and_return_conditional_losses_46856
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
:         Ат	2
	transposeЛ
"stft_1_tf.signal.stft/frame_lengthConst*
_output_shapes
: *
dtype0*
value
B :А2$
"stft_1_tf.signal.stft/frame_lengthЗ
 stft_1_tf.signal.stft/frame_stepConst*
_output_shapes
: *
dtype0*
value
B :А2"
 stft_1_tf.signal.stft/frame_stepЗ
 stft_1_tf.signal.stft/fft_lengthConst*
_output_shapes
: *
dtype0*
value
B :А2"
 stft_1_tf.signal.stft/fft_lengthП
 stft_1_tf.signal.stft/frame/axisConst*
_output_shapes
: *
dtype0*
valueB :
         2"
 stft_1_tf.signal.stft/frame/axisГ
!stft_1_tf.signal.stft/frame/ShapeShapetranspose:y:0*
T0*
_output_shapes
:2#
!stft_1_tf.signal.stft/frame/ShapeЖ
 stft_1_tf.signal.stft/frame/RankConst*
_output_shapes
: *
dtype0*
value	B :2"
 stft_1_tf.signal.stft/frame/RankФ
'stft_1_tf.signal.stft/frame/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2)
'stft_1_tf.signal.stft/frame/range/startФ
'stft_1_tf.signal.stft/frame/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2)
'stft_1_tf.signal.stft/frame/range/delta·
!stft_1_tf.signal.stft/frame/rangeRange0stft_1_tf.signal.stft/frame/range/start:output:0)stft_1_tf.signal.stft/frame/Rank:output:00stft_1_tf.signal.stft/frame/range/delta:output:0*
_output_shapes
:2#
!stft_1_tf.signal.stft/frame/range╡
/stft_1_tf.signal.stft/frame/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
         21
/stft_1_tf.signal.stft/frame/strided_slice/stack░
1stft_1_tf.signal.stft/frame/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 23
1stft_1_tf.signal.stft/frame/strided_slice/stack_1░
1stft_1_tf.signal.stft/frame/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1stft_1_tf.signal.stft/frame/strided_slice/stack_2К
)stft_1_tf.signal.stft/frame/strided_sliceStridedSlice*stft_1_tf.signal.stft/frame/range:output:08stft_1_tf.signal.stft/frame/strided_slice/stack:output:0:stft_1_tf.signal.stft/frame/strided_slice/stack_1:output:0:stft_1_tf.signal.stft/frame/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)stft_1_tf.signal.stft/frame/strided_sliceИ
!stft_1_tf.signal.stft/frame/sub/yConst*
_output_shapes
: *
dtype0*
value	B :2#
!stft_1_tf.signal.stft/frame/sub/y┴
stft_1_tf.signal.stft/frame/subSub)stft_1_tf.signal.stft/frame/Rank:output:0*stft_1_tf.signal.stft/frame/sub/y:output:0*
T0*
_output_shapes
: 2!
stft_1_tf.signal.stft/frame/sub╟
!stft_1_tf.signal.stft/frame/sub_1Sub#stft_1_tf.signal.stft/frame/sub:z:02stft_1_tf.signal.stft/frame/strided_slice:output:0*
T0*
_output_shapes
: 2#
!stft_1_tf.signal.stft/frame/sub_1О
$stft_1_tf.signal.stft/frame/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2&
$stft_1_tf.signal.stft/frame/packed/1И
"stft_1_tf.signal.stft/frame/packedPack2stft_1_tf.signal.stft/frame/strided_slice:output:0-stft_1_tf.signal.stft/frame/packed/1:output:0%stft_1_tf.signal.stft/frame/sub_1:z:0*
N*
T0*
_output_shapes
:2$
"stft_1_tf.signal.stft/frame/packedЬ
+stft_1_tf.signal.stft/frame/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2-
+stft_1_tf.signal.stft/frame/split/split_dimл
!stft_1_tf.signal.stft/frame/splitSplitV*stft_1_tf.signal.stft/frame/Shape:output:0+stft_1_tf.signal.stft/frame/packed:output:04stft_1_tf.signal.stft/frame/split/split_dim:output:0*
T0*

Tlen0*$
_output_shapes
::: *
	num_split2#
!stft_1_tf.signal.stft/frame/splitЩ
)stft_1_tf.signal.stft/frame/Reshape/shapeConst*
_output_shapes
: *
dtype0*
valueB 2+
)stft_1_tf.signal.stft/frame/Reshape/shapeЭ
+stft_1_tf.signal.stft/frame/Reshape/shape_1Const*
_output_shapes
: *
dtype0*
valueB 2-
+stft_1_tf.signal.stft/frame/Reshape/shape_1╪
#stft_1_tf.signal.stft/frame/ReshapeReshape*stft_1_tf.signal.stft/frame/split:output:14stft_1_tf.signal.stft/frame/Reshape/shape_1:output:0*
T0*
_output_shapes
: 2%
#stft_1_tf.signal.stft/frame/ReshapeЖ
 stft_1_tf.signal.stft/frame/SizeConst*
_output_shapes
: *
dtype0*
value	B :2"
 stft_1_tf.signal.stft/frame/SizeК
"stft_1_tf.signal.stft/frame/Size_1Const*
_output_shapes
: *
dtype0*
value	B : 2$
"stft_1_tf.signal.stft/frame/Size_1╔
!stft_1_tf.signal.stft/frame/sub_2Sub,stft_1_tf.signal.stft/frame/Reshape:output:0+stft_1_tf.signal.stft/frame_length:output:0*
T0*
_output_shapes
: 2#
!stft_1_tf.signal.stft/frame/sub_2╦
$stft_1_tf.signal.stft/frame/floordivFloorDiv%stft_1_tf.signal.stft/frame/sub_2:z:0)stft_1_tf.signal.stft/frame_step:output:0*
T0*
_output_shapes
: 2&
$stft_1_tf.signal.stft/frame/floordivИ
!stft_1_tf.signal.stft/frame/add/xConst*
_output_shapes
: *
dtype0*
value	B :2#
!stft_1_tf.signal.stft/frame/add/x┬
stft_1_tf.signal.stft/frame/addAddV2*stft_1_tf.signal.stft/frame/add/x:output:0(stft_1_tf.signal.stft/frame/floordiv:z:0*
T0*
_output_shapes
: 2!
stft_1_tf.signal.stft/frame/addР
%stft_1_tf.signal.stft/frame/Maximum/xConst*
_output_shapes
: *
dtype0*
value	B : 2'
%stft_1_tf.signal.stft/frame/Maximum/x╦
#stft_1_tf.signal.stft/frame/MaximumMaximum.stft_1_tf.signal.stft/frame/Maximum/x:output:0#stft_1_tf.signal.stft/frame/add:z:0*
T0*
_output_shapes
: 2%
#stft_1_tf.signal.stft/frame/MaximumС
%stft_1_tf.signal.stft/frame/gcd/ConstConst*
_output_shapes
: *
dtype0*
value
B :А2'
%stft_1_tf.signal.stft/frame/gcd/ConstЧ
(stft_1_tf.signal.stft/frame/floordiv_1/yConst*
_output_shapes
: *
dtype0*
value
B :А2*
(stft_1_tf.signal.stft/frame/floordiv_1/y▌
&stft_1_tf.signal.stft/frame/floordiv_1FloorDiv+stft_1_tf.signal.stft/frame_length:output:01stft_1_tf.signal.stft/frame/floordiv_1/y:output:0*
T0*
_output_shapes
: 2(
&stft_1_tf.signal.stft/frame/floordiv_1Ч
(stft_1_tf.signal.stft/frame/floordiv_2/yConst*
_output_shapes
: *
dtype0*
value
B :А2*
(stft_1_tf.signal.stft/frame/floordiv_2/y█
&stft_1_tf.signal.stft/frame/floordiv_2FloorDiv)stft_1_tf.signal.stft/frame_step:output:01stft_1_tf.signal.stft/frame/floordiv_2/y:output:0*
T0*
_output_shapes
: 2(
&stft_1_tf.signal.stft/frame/floordiv_2Ч
(stft_1_tf.signal.stft/frame/floordiv_3/yConst*
_output_shapes
: *
dtype0*
value
B :А2*
(stft_1_tf.signal.stft/frame/floordiv_3/y▐
&stft_1_tf.signal.stft/frame/floordiv_3FloorDiv,stft_1_tf.signal.stft/frame/Reshape:output:01stft_1_tf.signal.stft/frame/floordiv_3/y:output:0*
T0*
_output_shapes
: 2(
&stft_1_tf.signal.stft/frame/floordiv_3Й
!stft_1_tf.signal.stft/frame/mul/yConst*
_output_shapes
: *
dtype0*
value
B :А2#
!stft_1_tf.signal.stft/frame/mul/y┬
stft_1_tf.signal.stft/frame/mulMul*stft_1_tf.signal.stft/frame/floordiv_3:z:0*stft_1_tf.signal.stft/frame/mul/y:output:0*
T0*
_output_shapes
: 2!
stft_1_tf.signal.stft/frame/mul╡
+stft_1_tf.signal.stft/frame/concat/values_1Pack#stft_1_tf.signal.stft/frame/mul:z:0*
N*
T0*
_output_shapes
:2-
+stft_1_tf.signal.stft/frame/concat/values_1Ф
'stft_1_tf.signal.stft/frame/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2)
'stft_1_tf.signal.stft/frame/concat/axis┬
"stft_1_tf.signal.stft/frame/concatConcatV2*stft_1_tf.signal.stft/frame/split:output:04stft_1_tf.signal.stft/frame/concat/values_1:output:0*stft_1_tf.signal.stft/frame/split:output:20stft_1_tf.signal.stft/frame/concat/axis:output:0*
N*
T0*
_output_shapes
:2$
"stft_1_tf.signal.stft/frame/concatе
/stft_1_tf.signal.stft/frame/concat_1/values_1/1Const*
_output_shapes
: *
dtype0*
value
B :А21
/stft_1_tf.signal.stft/frame/concat_1/values_1/1·
-stft_1_tf.signal.stft/frame/concat_1/values_1Pack*stft_1_tf.signal.stft/frame/floordiv_3:z:08stft_1_tf.signal.stft/frame/concat_1/values_1/1:output:0*
N*
T0*
_output_shapes
:2/
-stft_1_tf.signal.stft/frame/concat_1/values_1Ш
)stft_1_tf.signal.stft/frame/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2+
)stft_1_tf.signal.stft/frame/concat_1/axis╩
$stft_1_tf.signal.stft/frame/concat_1ConcatV2*stft_1_tf.signal.stft/frame/split:output:06stft_1_tf.signal.stft/frame/concat_1/values_1:output:0*stft_1_tf.signal.stft/frame/split:output:22stft_1_tf.signal.stft/frame/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2&
$stft_1_tf.signal.stft/frame/concat_1Ъ
&stft_1_tf.signal.stft/frame/zeros_likeConst*
_output_shapes
:*
dtype0*
valueB: 2(
&stft_1_tf.signal.stft/frame/zeros_likeд
+stft_1_tf.signal.stft/frame/ones_like/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2-
+stft_1_tf.signal.stft/frame/ones_like/ShapeЬ
+stft_1_tf.signal.stft/frame/ones_like/ConstConst*
_output_shapes
: *
dtype0*
value	B :2-
+stft_1_tf.signal.stft/frame/ones_like/Constч
%stft_1_tf.signal.stft/frame/ones_likeFill4stft_1_tf.signal.stft/frame/ones_like/Shape:output:04stft_1_tf.signal.stft/frame/ones_like/Const:output:0*
T0*
_output_shapes
:2'
%stft_1_tf.signal.stft/frame/ones_like╓
(stft_1_tf.signal.stft/frame/StridedSliceStridedSlicetranspose:y:0/stft_1_tf.signal.stft/frame/zeros_like:output:0+stft_1_tf.signal.stft/frame/concat:output:0.stft_1_tf.signal.stft/frame/ones_like:output:0*
Index0*
T0*=
_output_shapes+
):'                           2*
(stft_1_tf.signal.stft/frame/StridedSliceИ
%stft_1_tf.signal.stft/frame/Reshape_1Reshape1stft_1_tf.signal.stft/frame/StridedSlice:output:0-stft_1_tf.signal.stft/frame/concat_1:output:0*
T0*B
_output_shapes0
.:,                           А2'
%stft_1_tf.signal.stft/frame/Reshape_1Ш
)stft_1_tf.signal.stft/frame/range_1/startConst*
_output_shapes
: *
dtype0*
value	B : 2+
)stft_1_tf.signal.stft/frame/range_1/startШ
)stft_1_tf.signal.stft/frame/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :2+
)stft_1_tf.signal.stft/frame/range_1/deltaЙ
#stft_1_tf.signal.stft/frame/range_1Range2stft_1_tf.signal.stft/frame/range_1/start:output:0'stft_1_tf.signal.stft/frame/Maximum:z:02stft_1_tf.signal.stft/frame/range_1/delta:output:0*#
_output_shapes
:         2%
#stft_1_tf.signal.stft/frame/range_1╒
!stft_1_tf.signal.stft/frame/mul_1Mul,stft_1_tf.signal.stft/frame/range_1:output:0*stft_1_tf.signal.stft/frame/floordiv_2:z:0*
T0*#
_output_shapes
:         2#
!stft_1_tf.signal.stft/frame/mul_1а
-stft_1_tf.signal.stft/frame/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2/
-stft_1_tf.signal.stft/frame/Reshape_2/shape/1ё
+stft_1_tf.signal.stft/frame/Reshape_2/shapePack'stft_1_tf.signal.stft/frame/Maximum:z:06stft_1_tf.signal.stft/frame/Reshape_2/shape/1:output:0*
N*
T0*
_output_shapes
:2-
+stft_1_tf.signal.stft/frame/Reshape_2/shapeш
%stft_1_tf.signal.stft/frame/Reshape_2Reshape%stft_1_tf.signal.stft/frame/mul_1:z:04stft_1_tf.signal.stft/frame/Reshape_2/shape:output:0*
T0*'
_output_shapes
:         2'
%stft_1_tf.signal.stft/frame/Reshape_2Ш
)stft_1_tf.signal.stft/frame/range_2/startConst*
_output_shapes
: *
dtype0*
value	B : 2+
)stft_1_tf.signal.stft/frame/range_2/startШ
)stft_1_tf.signal.stft/frame/range_2/deltaConst*
_output_shapes
: *
dtype0*
value	B :2+
)stft_1_tf.signal.stft/frame/range_2/deltaГ
#stft_1_tf.signal.stft/frame/range_2Range2stft_1_tf.signal.stft/frame/range_2/start:output:0*stft_1_tf.signal.stft/frame/floordiv_1:z:02stft_1_tf.signal.stft/frame/range_2/delta:output:0*
_output_shapes
:2%
#stft_1_tf.signal.stft/frame/range_2а
-stft_1_tf.signal.stft/frame/Reshape_3/shape/0Const*
_output_shapes
: *
dtype0*
value	B :2/
-stft_1_tf.signal.stft/frame/Reshape_3/shape/0Ї
+stft_1_tf.signal.stft/frame/Reshape_3/shapePack6stft_1_tf.signal.stft/frame/Reshape_3/shape/0:output:0*stft_1_tf.signal.stft/frame/floordiv_1:z:0*
N*
T0*
_output_shapes
:2-
+stft_1_tf.signal.stft/frame/Reshape_3/shapeц
%stft_1_tf.signal.stft/frame/Reshape_3Reshape,stft_1_tf.signal.stft/frame/range_2:output:04stft_1_tf.signal.stft/frame/Reshape_3/shape:output:0*
T0*
_output_shapes

:2'
%stft_1_tf.signal.stft/frame/Reshape_3с
!stft_1_tf.signal.stft/frame/add_1AddV2.stft_1_tf.signal.stft/frame/Reshape_2:output:0.stft_1_tf.signal.stft/frame/Reshape_3:output:0*
T0*'
_output_shapes
:         2#
!stft_1_tf.signal.stft/frame/add_1╫
$stft_1_tf.signal.stft/frame/GatherV2GatherV2.stft_1_tf.signal.stft/frame/Reshape_1:output:0%stft_1_tf.signal.stft/frame/add_1:z:02stft_1_tf.signal.stft/frame/strided_slice:output:0*
Taxis0*
Tindices0*
Tparams0*F
_output_shapes4
2:0                           А2&
$stft_1_tf.signal.stft/frame/GatherV2ъ
-stft_1_tf.signal.stft/frame/concat_2/values_1Pack'stft_1_tf.signal.stft/frame/Maximum:z:0+stft_1_tf.signal.stft/frame_length:output:0*
N*
T0*
_output_shapes
:2/
-stft_1_tf.signal.stft/frame/concat_2/values_1Ш
)stft_1_tf.signal.stft/frame/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2+
)stft_1_tf.signal.stft/frame/concat_2/axis╩
$stft_1_tf.signal.stft/frame/concat_2ConcatV2*stft_1_tf.signal.stft/frame/split:output:06stft_1_tf.signal.stft/frame/concat_2/values_1:output:0*stft_1_tf.signal.stft/frame/split:output:22stft_1_tf.signal.stft/frame/concat_2/axis:output:0*
N*
T0*
_output_shapes
:2&
$stft_1_tf.signal.stft/frame/concat_2є
%stft_1_tf.signal.stft/frame/Reshape_4Reshape-stft_1_tf.signal.stft/frame/GatherV2:output:0-stft_1_tf.signal.stft/frame/concat_2:output:0*
T0*1
_output_shapes
:         ╖А2'
%stft_1_tf.signal.stft/frame/Reshape_4Ъ
*stft_1_tf.signal.stft/hann_window/periodicConst*
_output_shapes
: *
dtype0
*
value	B
 Z2,
*stft_1_tf.signal.stft/hann_window/periodic╜
&stft_1_tf.signal.stft/hann_window/CastCast3stft_1_tf.signal.stft/hann_window/periodic:output:0*

DstT0*

SrcT0
*
_output_shapes
: 2(
&stft_1_tf.signal.stft/hann_window/CastЮ
,stft_1_tf.signal.stft/hann_window/FloorMod/yConst*
_output_shapes
: *
dtype0*
value	B :2.
,stft_1_tf.signal.stft/hann_window/FloorMod/yщ
*stft_1_tf.signal.stft/hann_window/FloorModFloorMod+stft_1_tf.signal.stft/frame_length:output:05stft_1_tf.signal.stft/hann_window/FloorMod/y:output:0*
T0*
_output_shapes
: 2,
*stft_1_tf.signal.stft/hann_window/FloorModФ
'stft_1_tf.signal.stft/hann_window/sub/xConst*
_output_shapes
: *
dtype0*
value	B :2)
'stft_1_tf.signal.stft/hann_window/sub/x╪
%stft_1_tf.signal.stft/hann_window/subSub0stft_1_tf.signal.stft/hann_window/sub/x:output:0.stft_1_tf.signal.stft/hann_window/FloorMod:z:0*
T0*
_output_shapes
: 2'
%stft_1_tf.signal.stft/hann_window/sub═
%stft_1_tf.signal.stft/hann_window/mulMul*stft_1_tf.signal.stft/hann_window/Cast:y:0)stft_1_tf.signal.stft/hann_window/sub:z:0*
T0*
_output_shapes
: 2'
%stft_1_tf.signal.stft/hann_window/mul╨
%stft_1_tf.signal.stft/hann_window/addAddV2+stft_1_tf.signal.stft/frame_length:output:0)stft_1_tf.signal.stft/hann_window/mul:z:0*
T0*
_output_shapes
: 2'
%stft_1_tf.signal.stft/hann_window/addШ
)stft_1_tf.signal.stft/hann_window/sub_1/yConst*
_output_shapes
: *
dtype0*
value	B :2+
)stft_1_tf.signal.stft/hann_window/sub_1/y┘
'stft_1_tf.signal.stft/hann_window/sub_1Sub)stft_1_tf.signal.stft/hann_window/add:z:02stft_1_tf.signal.stft/hann_window/sub_1/y:output:0*
T0*
_output_shapes
: 2)
'stft_1_tf.signal.stft/hann_window/sub_1╣
(stft_1_tf.signal.stft/hann_window/Cast_1Cast+stft_1_tf.signal.stft/hann_window/sub_1:z:0*

DstT0*

SrcT0*
_output_shapes
: 2*
(stft_1_tf.signal.stft/hann_window/Cast_1а
-stft_1_tf.signal.stft/hann_window/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2/
-stft_1_tf.signal.stft/hann_window/range/startа
-stft_1_tf.signal.stft/hann_window/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2/
-stft_1_tf.signal.stft/hann_window/range/deltaХ
'stft_1_tf.signal.stft/hann_window/rangeRange6stft_1_tf.signal.stft/hann_window/range/start:output:0+stft_1_tf.signal.stft/frame_length:output:06stft_1_tf.signal.stft/hann_window/range/delta:output:0*
_output_shapes	
:А2)
'stft_1_tf.signal.stft/hann_window/range├
(stft_1_tf.signal.stft/hann_window/Cast_2Cast0stft_1_tf.signal.stft/hann_window/range:output:0*

DstT0*

SrcT0*
_output_shapes	
:А2*
(stft_1_tf.signal.stft/hann_window/Cast_2Ч
'stft_1_tf.signal.stft/hann_window/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *█╔@2)
'stft_1_tf.signal.stft/hann_window/Const▀
'stft_1_tf.signal.stft/hann_window/mul_1Mul0stft_1_tf.signal.stft/hann_window/Const:output:0,stft_1_tf.signal.stft/hann_window/Cast_2:y:0*
T0*
_output_shapes	
:А2)
'stft_1_tf.signal.stft/hann_window/mul_1т
)stft_1_tf.signal.stft/hann_window/truedivRealDiv+stft_1_tf.signal.stft/hann_window/mul_1:z:0,stft_1_tf.signal.stft/hann_window/Cast_1:y:0*
T0*
_output_shapes	
:А2+
)stft_1_tf.signal.stft/hann_window/truedivк
%stft_1_tf.signal.stft/hann_window/CosCos-stft_1_tf.signal.stft/hann_window/truediv:z:0*
T0*
_output_shapes	
:А2'
%stft_1_tf.signal.stft/hann_window/CosЫ
)stft_1_tf.signal.stft/hann_window/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2+
)stft_1_tf.signal.stft/hann_window/mul_2/x▐
'stft_1_tf.signal.stft/hann_window/mul_2Mul2stft_1_tf.signal.stft/hann_window/mul_2/x:output:0)stft_1_tf.signal.stft/hann_window/Cos:y:0*
T0*
_output_shapes	
:А2)
'stft_1_tf.signal.stft/hann_window/mul_2Ы
)stft_1_tf.signal.stft/hann_window/sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2+
)stft_1_tf.signal.stft/hann_window/sub_2/xр
'stft_1_tf.signal.stft/hann_window/sub_2Sub2stft_1_tf.signal.stft/hann_window/sub_2/x:output:0+stft_1_tf.signal.stft/hann_window/mul_2:z:0*
T0*
_output_shapes	
:А2)
'stft_1_tf.signal.stft/hann_window/sub_2╓
stft_1_tf.signal.stft/mulMul.stft_1_tf.signal.stft/frame/Reshape_4:output:0+stft_1_tf.signal.stft/hann_window/sub_2:z:0*
T0*1
_output_shapes
:         ╖А2
stft_1_tf.signal.stft/mulз
!stft_1_tf.signal.stft/rfft/packedPack)stft_1_tf.signal.stft/fft_length:output:0*
N*
T0*
_output_shapes
:2#
!stft_1_tf.signal.stft/rfft/packedЩ
%stft_1_tf.signal.stft/rfft/fft_lengthConst*
_output_shapes
:*
dtype0*
valueB:А2'
%stft_1_tf.signal.stft/rfft/fft_length┬
stft_1_tf.signal.stft/rfftRFFTstft_1_tf.signal.stft/mul:z:0.stft_1_tf.signal.stft/rfft/fft_length:output:0*1
_output_shapes
:         ╖Б2
stft_1_tf.signal.stft/rfft}
transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             2
transpose_1/permг
transpose_1	Transpose#stft_1_tf.signal.stft/rfft:output:0transpose_1/perm:output:0*
T0*1
_output_shapes
:         ╖Б2
transpose_1m
IdentityIdentitytranspose_1:y:0*
T0*1
_output_shapes
:         ╖Б2

Identity"
identityIdentity:output:0*,
_input_shapes
:         Ат	:P L
-
_output_shapes
:         Ат	

_user_specified_namex
╝
J
.__inference_stft_magnitude_layer_call_fn_46045

inputs
identity╤
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ╖Б* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_stft_magnitude_layer_call_and_return_conditional_losses_437482
PartitionedCallv
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:         ╖Б2

Identity"
identityIdentity:output:0*,
_input_shapes
:         Ат	:U Q
-
_output_shapes
:         Ат	
 
_user_specified_nameinputs
Ы
и
5__inference_batch_normalization_6_layer_call_fn_46397

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCall▓
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Y
fTRR
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_440562
StatefulPartitionedCallи
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
л
K
/__inference_max_pooling2d_5_layer_call_fn_44110

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
6:4                                    * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_441042
PartitionedCallП
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
└
є
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_44203

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1▄
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3ь
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                            ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
╙
и
5__inference_batch_normalization_7_layer_call_fn_46609

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCallа
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Y
fTRR
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_445722
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:          2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:          ::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:          
 
_user_specified_nameinputs
°
є
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_44489

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1╩
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         39:::::*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3┌
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:         392

Identity"
identityIdentity:output:0*>
_input_shapes-
+:         39::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:         39
 
_user_specified_nameinputs
ь
e
I__inference_stft_magnitude_layer_call_and_return_conditional_losses_43748

inputs
identity╫
stft_1/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ╖Б* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_stft_1_layer_call_and_return_conditional_losses_437112
stft_1/PartitionedCall 
magnitude_1/PartitionedCallPartitionedCallstft_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ╖Б* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_magnitude_1_layer_call_and_return_conditional_losses_437242
magnitude_1/PartitionedCallВ
IdentityIdentity$magnitude_1/PartitionedCall:output:0*
T0*1
_output_shapes
:         ╖Б2

Identity"
identityIdentity:output:0*,
_input_shapes
:         Ат	:U Q
-
_output_shapes
:         Ат	
 
_user_specified_nameinputs
°
є
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_44590

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1╩
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:          : : : : :*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3┌
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:          2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:          ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:          
 
_user_specified_nameinputs
п
B
+__inference_magnitude_1_layer_call_fn_46871
x
identity╔
PartitionedCallPartitionedCallx*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ╖Б* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_magnitude_1_layer_call_and_return_conditional_losses_437242
PartitionedCallv
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:         ╖Б2

Identity"
identityIdentity:output:0*0
_input_shapes
:         ╖Б:T P
1
_output_shapes
:         ╖Б

_user_specified_namex
┘

▄
C__inference_conv2d_3_layer_call_and_return_conditional_losses_46189

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpе
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ╖Б*
paddingSAME*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpК
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ╖Б2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:         ╖Б2
Reluб
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:         ╖Б2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:         ╖Б::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:         ╖Б
 
_user_specified_nameinputs
└
є
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_43855

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1▄
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3ь
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
╙

▄
C__inference_conv2d_4_layer_call_and_return_conditional_losses_46337

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpд
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         gл*
paddingSAME*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpЙ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         gл2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:         gл2
Reluа
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:         gл2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:         gл::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:         gл
 
_user_specified_nameinputs
╨▓
e
I__inference_stft_magnitude_layer_call_and_return_conditional_losses_46040

inputs
identityГ
stft_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
stft_1/transpose/permС
stft_1/transpose	Transposeinputsstft_1/transpose/perm:output:0*
T0*-
_output_shapes
:         Ат	2
stft_1/transposeЩ
)stft_1/stft_1_tf.signal.stft/frame_lengthConst*
_output_shapes
: *
dtype0*
value
B :А2+
)stft_1/stft_1_tf.signal.stft/frame_lengthХ
'stft_1/stft_1_tf.signal.stft/frame_stepConst*
_output_shapes
: *
dtype0*
value
B :А2)
'stft_1/stft_1_tf.signal.stft/frame_stepХ
'stft_1/stft_1_tf.signal.stft/fft_lengthConst*
_output_shapes
: *
dtype0*
value
B :А2)
'stft_1/stft_1_tf.signal.stft/fft_lengthЭ
'stft_1/stft_1_tf.signal.stft/frame/axisConst*
_output_shapes
: *
dtype0*
valueB :
         2)
'stft_1/stft_1_tf.signal.stft/frame/axisШ
(stft_1/stft_1_tf.signal.stft/frame/ShapeShapestft_1/transpose:y:0*
T0*
_output_shapes
:2*
(stft_1/stft_1_tf.signal.stft/frame/ShapeФ
'stft_1/stft_1_tf.signal.stft/frame/RankConst*
_output_shapes
: *
dtype0*
value	B :2)
'stft_1/stft_1_tf.signal.stft/frame/Rankв
.stft_1/stft_1_tf.signal.stft/frame/range/startConst*
_output_shapes
: *
dtype0*
value	B : 20
.stft_1/stft_1_tf.signal.stft/frame/range/startв
.stft_1/stft_1_tf.signal.stft/frame/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :20
.stft_1/stft_1_tf.signal.stft/frame/range/deltaЭ
(stft_1/stft_1_tf.signal.stft/frame/rangeRange7stft_1/stft_1_tf.signal.stft/frame/range/start:output:00stft_1/stft_1_tf.signal.stft/frame/Rank:output:07stft_1/stft_1_tf.signal.stft/frame/range/delta:output:0*
_output_shapes
:2*
(stft_1/stft_1_tf.signal.stft/frame/range├
6stft_1/stft_1_tf.signal.stft/frame/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
         28
6stft_1/stft_1_tf.signal.stft/frame/strided_slice/stack╛
8stft_1/stft_1_tf.signal.stft/frame/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2:
8stft_1/stft_1_tf.signal.stft/frame/strided_slice/stack_1╛
8stft_1/stft_1_tf.signal.stft/frame/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2:
8stft_1/stft_1_tf.signal.stft/frame/strided_slice/stack_2┤
0stft_1/stft_1_tf.signal.stft/frame/strided_sliceStridedSlice1stft_1/stft_1_tf.signal.stft/frame/range:output:0?stft_1/stft_1_tf.signal.stft/frame/strided_slice/stack:output:0Astft_1/stft_1_tf.signal.stft/frame/strided_slice/stack_1:output:0Astft_1/stft_1_tf.signal.stft/frame/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask22
0stft_1/stft_1_tf.signal.stft/frame/strided_sliceЦ
(stft_1/stft_1_tf.signal.stft/frame/sub/yConst*
_output_shapes
: *
dtype0*
value	B :2*
(stft_1/stft_1_tf.signal.stft/frame/sub/y▌
&stft_1/stft_1_tf.signal.stft/frame/subSub0stft_1/stft_1_tf.signal.stft/frame/Rank:output:01stft_1/stft_1_tf.signal.stft/frame/sub/y:output:0*
T0*
_output_shapes
: 2(
&stft_1/stft_1_tf.signal.stft/frame/subу
(stft_1/stft_1_tf.signal.stft/frame/sub_1Sub*stft_1/stft_1_tf.signal.stft/frame/sub:z:09stft_1/stft_1_tf.signal.stft/frame/strided_slice:output:0*
T0*
_output_shapes
: 2*
(stft_1/stft_1_tf.signal.stft/frame/sub_1Ь
+stft_1/stft_1_tf.signal.stft/frame/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2-
+stft_1/stft_1_tf.signal.stft/frame/packed/1л
)stft_1/stft_1_tf.signal.stft/frame/packedPack9stft_1/stft_1_tf.signal.stft/frame/strided_slice:output:04stft_1/stft_1_tf.signal.stft/frame/packed/1:output:0,stft_1/stft_1_tf.signal.stft/frame/sub_1:z:0*
N*
T0*
_output_shapes
:2+
)stft_1/stft_1_tf.signal.stft/frame/packedк
2stft_1/stft_1_tf.signal.stft/frame/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 24
2stft_1/stft_1_tf.signal.stft/frame/split/split_dim╬
(stft_1/stft_1_tf.signal.stft/frame/splitSplitV1stft_1/stft_1_tf.signal.stft/frame/Shape:output:02stft_1/stft_1_tf.signal.stft/frame/packed:output:0;stft_1/stft_1_tf.signal.stft/frame/split/split_dim:output:0*
T0*

Tlen0*$
_output_shapes
::: *
	num_split2*
(stft_1/stft_1_tf.signal.stft/frame/splitз
0stft_1/stft_1_tf.signal.stft/frame/Reshape/shapeConst*
_output_shapes
: *
dtype0*
valueB 22
0stft_1/stft_1_tf.signal.stft/frame/Reshape/shapeл
2stft_1/stft_1_tf.signal.stft/frame/Reshape/shape_1Const*
_output_shapes
: *
dtype0*
valueB 24
2stft_1/stft_1_tf.signal.stft/frame/Reshape/shape_1Ї
*stft_1/stft_1_tf.signal.stft/frame/ReshapeReshape1stft_1/stft_1_tf.signal.stft/frame/split:output:1;stft_1/stft_1_tf.signal.stft/frame/Reshape/shape_1:output:0*
T0*
_output_shapes
: 2,
*stft_1/stft_1_tf.signal.stft/frame/ReshapeФ
'stft_1/stft_1_tf.signal.stft/frame/SizeConst*
_output_shapes
: *
dtype0*
value	B :2)
'stft_1/stft_1_tf.signal.stft/frame/SizeШ
)stft_1/stft_1_tf.signal.stft/frame/Size_1Const*
_output_shapes
: *
dtype0*
value	B : 2+
)stft_1/stft_1_tf.signal.stft/frame/Size_1х
(stft_1/stft_1_tf.signal.stft/frame/sub_2Sub3stft_1/stft_1_tf.signal.stft/frame/Reshape:output:02stft_1/stft_1_tf.signal.stft/frame_length:output:0*
T0*
_output_shapes
: 2*
(stft_1/stft_1_tf.signal.stft/frame/sub_2ч
+stft_1/stft_1_tf.signal.stft/frame/floordivFloorDiv,stft_1/stft_1_tf.signal.stft/frame/sub_2:z:00stft_1/stft_1_tf.signal.stft/frame_step:output:0*
T0*
_output_shapes
: 2-
+stft_1/stft_1_tf.signal.stft/frame/floordivЦ
(stft_1/stft_1_tf.signal.stft/frame/add/xConst*
_output_shapes
: *
dtype0*
value	B :2*
(stft_1/stft_1_tf.signal.stft/frame/add/x▐
&stft_1/stft_1_tf.signal.stft/frame/addAddV21stft_1/stft_1_tf.signal.stft/frame/add/x:output:0/stft_1/stft_1_tf.signal.stft/frame/floordiv:z:0*
T0*
_output_shapes
: 2(
&stft_1/stft_1_tf.signal.stft/frame/addЮ
,stft_1/stft_1_tf.signal.stft/frame/Maximum/xConst*
_output_shapes
: *
dtype0*
value	B : 2.
,stft_1/stft_1_tf.signal.stft/frame/Maximum/xч
*stft_1/stft_1_tf.signal.stft/frame/MaximumMaximum5stft_1/stft_1_tf.signal.stft/frame/Maximum/x:output:0*stft_1/stft_1_tf.signal.stft/frame/add:z:0*
T0*
_output_shapes
: 2,
*stft_1/stft_1_tf.signal.stft/frame/MaximumЯ
,stft_1/stft_1_tf.signal.stft/frame/gcd/ConstConst*
_output_shapes
: *
dtype0*
value
B :А2.
,stft_1/stft_1_tf.signal.stft/frame/gcd/Constе
/stft_1/stft_1_tf.signal.stft/frame/floordiv_1/yConst*
_output_shapes
: *
dtype0*
value
B :А21
/stft_1/stft_1_tf.signal.stft/frame/floordiv_1/y∙
-stft_1/stft_1_tf.signal.stft/frame/floordiv_1FloorDiv2stft_1/stft_1_tf.signal.stft/frame_length:output:08stft_1/stft_1_tf.signal.stft/frame/floordiv_1/y:output:0*
T0*
_output_shapes
: 2/
-stft_1/stft_1_tf.signal.stft/frame/floordiv_1е
/stft_1/stft_1_tf.signal.stft/frame/floordiv_2/yConst*
_output_shapes
: *
dtype0*
value
B :А21
/stft_1/stft_1_tf.signal.stft/frame/floordiv_2/yў
-stft_1/stft_1_tf.signal.stft/frame/floordiv_2FloorDiv0stft_1/stft_1_tf.signal.stft/frame_step:output:08stft_1/stft_1_tf.signal.stft/frame/floordiv_2/y:output:0*
T0*
_output_shapes
: 2/
-stft_1/stft_1_tf.signal.stft/frame/floordiv_2е
/stft_1/stft_1_tf.signal.stft/frame/floordiv_3/yConst*
_output_shapes
: *
dtype0*
value
B :А21
/stft_1/stft_1_tf.signal.stft/frame/floordiv_3/y·
-stft_1/stft_1_tf.signal.stft/frame/floordiv_3FloorDiv3stft_1/stft_1_tf.signal.stft/frame/Reshape:output:08stft_1/stft_1_tf.signal.stft/frame/floordiv_3/y:output:0*
T0*
_output_shapes
: 2/
-stft_1/stft_1_tf.signal.stft/frame/floordiv_3Ч
(stft_1/stft_1_tf.signal.stft/frame/mul/yConst*
_output_shapes
: *
dtype0*
value
B :А2*
(stft_1/stft_1_tf.signal.stft/frame/mul/y▐
&stft_1/stft_1_tf.signal.stft/frame/mulMul1stft_1/stft_1_tf.signal.stft/frame/floordiv_3:z:01stft_1/stft_1_tf.signal.stft/frame/mul/y:output:0*
T0*
_output_shapes
: 2(
&stft_1/stft_1_tf.signal.stft/frame/mul╩
2stft_1/stft_1_tf.signal.stft/frame/concat/values_1Pack*stft_1/stft_1_tf.signal.stft/frame/mul:z:0*
N*
T0*
_output_shapes
:24
2stft_1/stft_1_tf.signal.stft/frame/concat/values_1в
.stft_1/stft_1_tf.signal.stft/frame/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.stft_1/stft_1_tf.signal.stft/frame/concat/axisь
)stft_1/stft_1_tf.signal.stft/frame/concatConcatV21stft_1/stft_1_tf.signal.stft/frame/split:output:0;stft_1/stft_1_tf.signal.stft/frame/concat/values_1:output:01stft_1/stft_1_tf.signal.stft/frame/split:output:27stft_1/stft_1_tf.signal.stft/frame/concat/axis:output:0*
N*
T0*
_output_shapes
:2+
)stft_1/stft_1_tf.signal.stft/frame/concat│
6stft_1/stft_1_tf.signal.stft/frame/concat_1/values_1/1Const*
_output_shapes
: *
dtype0*
value
B :А28
6stft_1/stft_1_tf.signal.stft/frame/concat_1/values_1/1Ц
4stft_1/stft_1_tf.signal.stft/frame/concat_1/values_1Pack1stft_1/stft_1_tf.signal.stft/frame/floordiv_3:z:0?stft_1/stft_1_tf.signal.stft/frame/concat_1/values_1/1:output:0*
N*
T0*
_output_shapes
:26
4stft_1/stft_1_tf.signal.stft/frame/concat_1/values_1ж
0stft_1/stft_1_tf.signal.stft/frame/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 22
0stft_1/stft_1_tf.signal.stft/frame/concat_1/axisЇ
+stft_1/stft_1_tf.signal.stft/frame/concat_1ConcatV21stft_1/stft_1_tf.signal.stft/frame/split:output:0=stft_1/stft_1_tf.signal.stft/frame/concat_1/values_1:output:01stft_1/stft_1_tf.signal.stft/frame/split:output:29stft_1/stft_1_tf.signal.stft/frame/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2-
+stft_1/stft_1_tf.signal.stft/frame/concat_1и
-stft_1/stft_1_tf.signal.stft/frame/zeros_likeConst*
_output_shapes
:*
dtype0*
valueB: 2/
-stft_1/stft_1_tf.signal.stft/frame/zeros_like▓
2stft_1/stft_1_tf.signal.stft/frame/ones_like/ShapeConst*
_output_shapes
:*
dtype0*
valueB:24
2stft_1/stft_1_tf.signal.stft/frame/ones_like/Shapeк
2stft_1/stft_1_tf.signal.stft/frame/ones_like/ConstConst*
_output_shapes
: *
dtype0*
value	B :24
2stft_1/stft_1_tf.signal.stft/frame/ones_like/ConstГ
,stft_1/stft_1_tf.signal.stft/frame/ones_likeFill;stft_1/stft_1_tf.signal.stft/frame/ones_like/Shape:output:0;stft_1/stft_1_tf.signal.stft/frame/ones_like/Const:output:0*
T0*
_output_shapes
:2.
,stft_1/stft_1_tf.signal.stft/frame/ones_likeА
/stft_1/stft_1_tf.signal.stft/frame/StridedSliceStridedSlicestft_1/transpose:y:06stft_1/stft_1_tf.signal.stft/frame/zeros_like:output:02stft_1/stft_1_tf.signal.stft/frame/concat:output:05stft_1/stft_1_tf.signal.stft/frame/ones_like:output:0*
Index0*
T0*=
_output_shapes+
):'                           21
/stft_1/stft_1_tf.signal.stft/frame/StridedSliceд
,stft_1/stft_1_tf.signal.stft/frame/Reshape_1Reshape8stft_1/stft_1_tf.signal.stft/frame/StridedSlice:output:04stft_1/stft_1_tf.signal.stft/frame/concat_1:output:0*
T0*B
_output_shapes0
.:,                           А2.
,stft_1/stft_1_tf.signal.stft/frame/Reshape_1ж
0stft_1/stft_1_tf.signal.stft/frame/range_1/startConst*
_output_shapes
: *
dtype0*
value	B : 22
0stft_1/stft_1_tf.signal.stft/frame/range_1/startж
0stft_1/stft_1_tf.signal.stft/frame/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :22
0stft_1/stft_1_tf.signal.stft/frame/range_1/deltaм
*stft_1/stft_1_tf.signal.stft/frame/range_1Range9stft_1/stft_1_tf.signal.stft/frame/range_1/start:output:0.stft_1/stft_1_tf.signal.stft/frame/Maximum:z:09stft_1/stft_1_tf.signal.stft/frame/range_1/delta:output:0*#
_output_shapes
:         2,
*stft_1/stft_1_tf.signal.stft/frame/range_1ё
(stft_1/stft_1_tf.signal.stft/frame/mul_1Mul3stft_1/stft_1_tf.signal.stft/frame/range_1:output:01stft_1/stft_1_tf.signal.stft/frame/floordiv_2:z:0*
T0*#
_output_shapes
:         2*
(stft_1/stft_1_tf.signal.stft/frame/mul_1о
4stft_1/stft_1_tf.signal.stft/frame/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :26
4stft_1/stft_1_tf.signal.stft/frame/Reshape_2/shape/1Н
2stft_1/stft_1_tf.signal.stft/frame/Reshape_2/shapePack.stft_1/stft_1_tf.signal.stft/frame/Maximum:z:0=stft_1/stft_1_tf.signal.stft/frame/Reshape_2/shape/1:output:0*
N*
T0*
_output_shapes
:24
2stft_1/stft_1_tf.signal.stft/frame/Reshape_2/shapeД
,stft_1/stft_1_tf.signal.stft/frame/Reshape_2Reshape,stft_1/stft_1_tf.signal.stft/frame/mul_1:z:0;stft_1/stft_1_tf.signal.stft/frame/Reshape_2/shape:output:0*
T0*'
_output_shapes
:         2.
,stft_1/stft_1_tf.signal.stft/frame/Reshape_2ж
0stft_1/stft_1_tf.signal.stft/frame/range_2/startConst*
_output_shapes
: *
dtype0*
value	B : 22
0stft_1/stft_1_tf.signal.stft/frame/range_2/startж
0stft_1/stft_1_tf.signal.stft/frame/range_2/deltaConst*
_output_shapes
: *
dtype0*
value	B :22
0stft_1/stft_1_tf.signal.stft/frame/range_2/deltaж
*stft_1/stft_1_tf.signal.stft/frame/range_2Range9stft_1/stft_1_tf.signal.stft/frame/range_2/start:output:01stft_1/stft_1_tf.signal.stft/frame/floordiv_1:z:09stft_1/stft_1_tf.signal.stft/frame/range_2/delta:output:0*
_output_shapes
:2,
*stft_1/stft_1_tf.signal.stft/frame/range_2о
4stft_1/stft_1_tf.signal.stft/frame/Reshape_3/shape/0Const*
_output_shapes
: *
dtype0*
value	B :26
4stft_1/stft_1_tf.signal.stft/frame/Reshape_3/shape/0Р
2stft_1/stft_1_tf.signal.stft/frame/Reshape_3/shapePack=stft_1/stft_1_tf.signal.stft/frame/Reshape_3/shape/0:output:01stft_1/stft_1_tf.signal.stft/frame/floordiv_1:z:0*
N*
T0*
_output_shapes
:24
2stft_1/stft_1_tf.signal.stft/frame/Reshape_3/shapeВ
,stft_1/stft_1_tf.signal.stft/frame/Reshape_3Reshape3stft_1/stft_1_tf.signal.stft/frame/range_2:output:0;stft_1/stft_1_tf.signal.stft/frame/Reshape_3/shape:output:0*
T0*
_output_shapes

:2.
,stft_1/stft_1_tf.signal.stft/frame/Reshape_3¤
(stft_1/stft_1_tf.signal.stft/frame/add_1AddV25stft_1/stft_1_tf.signal.stft/frame/Reshape_2:output:05stft_1/stft_1_tf.signal.stft/frame/Reshape_3:output:0*
T0*'
_output_shapes
:         2*
(stft_1/stft_1_tf.signal.stft/frame/add_1·
+stft_1/stft_1_tf.signal.stft/frame/GatherV2GatherV25stft_1/stft_1_tf.signal.stft/frame/Reshape_1:output:0,stft_1/stft_1_tf.signal.stft/frame/add_1:z:09stft_1/stft_1_tf.signal.stft/frame/strided_slice:output:0*
Taxis0*
Tindices0*
Tparams0*F
_output_shapes4
2:0                           А2-
+stft_1/stft_1_tf.signal.stft/frame/GatherV2Ж
4stft_1/stft_1_tf.signal.stft/frame/concat_2/values_1Pack.stft_1/stft_1_tf.signal.stft/frame/Maximum:z:02stft_1/stft_1_tf.signal.stft/frame_length:output:0*
N*
T0*
_output_shapes
:26
4stft_1/stft_1_tf.signal.stft/frame/concat_2/values_1ж
0stft_1/stft_1_tf.signal.stft/frame/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 22
0stft_1/stft_1_tf.signal.stft/frame/concat_2/axisЇ
+stft_1/stft_1_tf.signal.stft/frame/concat_2ConcatV21stft_1/stft_1_tf.signal.stft/frame/split:output:0=stft_1/stft_1_tf.signal.stft/frame/concat_2/values_1:output:01stft_1/stft_1_tf.signal.stft/frame/split:output:29stft_1/stft_1_tf.signal.stft/frame/concat_2/axis:output:0*
N*
T0*
_output_shapes
:2-
+stft_1/stft_1_tf.signal.stft/frame/concat_2П
,stft_1/stft_1_tf.signal.stft/frame/Reshape_4Reshape4stft_1/stft_1_tf.signal.stft/frame/GatherV2:output:04stft_1/stft_1_tf.signal.stft/frame/concat_2:output:0*
T0*1
_output_shapes
:         ╖А2.
,stft_1/stft_1_tf.signal.stft/frame/Reshape_4и
1stft_1/stft_1_tf.signal.stft/hann_window/periodicConst*
_output_shapes
: *
dtype0
*
value	B
 Z23
1stft_1/stft_1_tf.signal.stft/hann_window/periodic╥
-stft_1/stft_1_tf.signal.stft/hann_window/CastCast:stft_1/stft_1_tf.signal.stft/hann_window/periodic:output:0*

DstT0*

SrcT0
*
_output_shapes
: 2/
-stft_1/stft_1_tf.signal.stft/hann_window/Castм
3stft_1/stft_1_tf.signal.stft/hann_window/FloorMod/yConst*
_output_shapes
: *
dtype0*
value	B :25
3stft_1/stft_1_tf.signal.stft/hann_window/FloorMod/yЕ
1stft_1/stft_1_tf.signal.stft/hann_window/FloorModFloorMod2stft_1/stft_1_tf.signal.stft/frame_length:output:0<stft_1/stft_1_tf.signal.stft/hann_window/FloorMod/y:output:0*
T0*
_output_shapes
: 23
1stft_1/stft_1_tf.signal.stft/hann_window/FloorModв
.stft_1/stft_1_tf.signal.stft/hann_window/sub/xConst*
_output_shapes
: *
dtype0*
value	B :20
.stft_1/stft_1_tf.signal.stft/hann_window/sub/xЇ
,stft_1/stft_1_tf.signal.stft/hann_window/subSub7stft_1/stft_1_tf.signal.stft/hann_window/sub/x:output:05stft_1/stft_1_tf.signal.stft/hann_window/FloorMod:z:0*
T0*
_output_shapes
: 2.
,stft_1/stft_1_tf.signal.stft/hann_window/subщ
,stft_1/stft_1_tf.signal.stft/hann_window/mulMul1stft_1/stft_1_tf.signal.stft/hann_window/Cast:y:00stft_1/stft_1_tf.signal.stft/hann_window/sub:z:0*
T0*
_output_shapes
: 2.
,stft_1/stft_1_tf.signal.stft/hann_window/mulь
,stft_1/stft_1_tf.signal.stft/hann_window/addAddV22stft_1/stft_1_tf.signal.stft/frame_length:output:00stft_1/stft_1_tf.signal.stft/hann_window/mul:z:0*
T0*
_output_shapes
: 2.
,stft_1/stft_1_tf.signal.stft/hann_window/addж
0stft_1/stft_1_tf.signal.stft/hann_window/sub_1/yConst*
_output_shapes
: *
dtype0*
value	B :22
0stft_1/stft_1_tf.signal.stft/hann_window/sub_1/yї
.stft_1/stft_1_tf.signal.stft/hann_window/sub_1Sub0stft_1/stft_1_tf.signal.stft/hann_window/add:z:09stft_1/stft_1_tf.signal.stft/hann_window/sub_1/y:output:0*
T0*
_output_shapes
: 20
.stft_1/stft_1_tf.signal.stft/hann_window/sub_1╬
/stft_1/stft_1_tf.signal.stft/hann_window/Cast_1Cast2stft_1/stft_1_tf.signal.stft/hann_window/sub_1:z:0*

DstT0*

SrcT0*
_output_shapes
: 21
/stft_1/stft_1_tf.signal.stft/hann_window/Cast_1о
4stft_1/stft_1_tf.signal.stft/hann_window/range/startConst*
_output_shapes
: *
dtype0*
value	B : 26
4stft_1/stft_1_tf.signal.stft/hann_window/range/startо
4stft_1/stft_1_tf.signal.stft/hann_window/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :26
4stft_1/stft_1_tf.signal.stft/hann_window/range/delta╕
.stft_1/stft_1_tf.signal.stft/hann_window/rangeRange=stft_1/stft_1_tf.signal.stft/hann_window/range/start:output:02stft_1/stft_1_tf.signal.stft/frame_length:output:0=stft_1/stft_1_tf.signal.stft/hann_window/range/delta:output:0*
_output_shapes	
:А20
.stft_1/stft_1_tf.signal.stft/hann_window/range╪
/stft_1/stft_1_tf.signal.stft/hann_window/Cast_2Cast7stft_1/stft_1_tf.signal.stft/hann_window/range:output:0*

DstT0*

SrcT0*
_output_shapes	
:А21
/stft_1/stft_1_tf.signal.stft/hann_window/Cast_2е
.stft_1/stft_1_tf.signal.stft/hann_window/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *█╔@20
.stft_1/stft_1_tf.signal.stft/hann_window/Const√
.stft_1/stft_1_tf.signal.stft/hann_window/mul_1Mul7stft_1/stft_1_tf.signal.stft/hann_window/Const:output:03stft_1/stft_1_tf.signal.stft/hann_window/Cast_2:y:0*
T0*
_output_shapes	
:А20
.stft_1/stft_1_tf.signal.stft/hann_window/mul_1■
0stft_1/stft_1_tf.signal.stft/hann_window/truedivRealDiv2stft_1/stft_1_tf.signal.stft/hann_window/mul_1:z:03stft_1/stft_1_tf.signal.stft/hann_window/Cast_1:y:0*
T0*
_output_shapes	
:А22
0stft_1/stft_1_tf.signal.stft/hann_window/truediv┐
,stft_1/stft_1_tf.signal.stft/hann_window/CosCos4stft_1/stft_1_tf.signal.stft/hann_window/truediv:z:0*
T0*
_output_shapes	
:А2.
,stft_1/stft_1_tf.signal.stft/hann_window/Cosй
0stft_1/stft_1_tf.signal.stft/hann_window/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?22
0stft_1/stft_1_tf.signal.stft/hann_window/mul_2/x·
.stft_1/stft_1_tf.signal.stft/hann_window/mul_2Mul9stft_1/stft_1_tf.signal.stft/hann_window/mul_2/x:output:00stft_1/stft_1_tf.signal.stft/hann_window/Cos:y:0*
T0*
_output_shapes	
:А20
.stft_1/stft_1_tf.signal.stft/hann_window/mul_2й
0stft_1/stft_1_tf.signal.stft/hann_window/sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?22
0stft_1/stft_1_tf.signal.stft/hann_window/sub_2/x№
.stft_1/stft_1_tf.signal.stft/hann_window/sub_2Sub9stft_1/stft_1_tf.signal.stft/hann_window/sub_2/x:output:02stft_1/stft_1_tf.signal.stft/hann_window/mul_2:z:0*
T0*
_output_shapes	
:А20
.stft_1/stft_1_tf.signal.stft/hann_window/sub_2Є
 stft_1/stft_1_tf.signal.stft/mulMul5stft_1/stft_1_tf.signal.stft/frame/Reshape_4:output:02stft_1/stft_1_tf.signal.stft/hann_window/sub_2:z:0*
T0*1
_output_shapes
:         ╖А2"
 stft_1/stft_1_tf.signal.stft/mul╝
(stft_1/stft_1_tf.signal.stft/rfft/packedPack0stft_1/stft_1_tf.signal.stft/fft_length:output:0*
N*
T0*
_output_shapes
:2*
(stft_1/stft_1_tf.signal.stft/rfft/packedз
,stft_1/stft_1_tf.signal.stft/rfft/fft_lengthConst*
_output_shapes
:*
dtype0*
valueB:А2.
,stft_1/stft_1_tf.signal.stft/rfft/fft_length▐
!stft_1/stft_1_tf.signal.stft/rfftRFFT$stft_1/stft_1_tf.signal.stft/mul:z:05stft_1/stft_1_tf.signal.stft/rfft/fft_length:output:0*1
_output_shapes
:         ╖Б2#
!stft_1/stft_1_tf.signal.stft/rfftЛ
stft_1/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             2
stft_1/transpose_1/perm┐
stft_1/transpose_1	Transpose*stft_1/stft_1_tf.signal.stft/rfft:output:0 stft_1/transpose_1/perm:output:0*
T0*1
_output_shapes
:         ╖Б2
stft_1/transpose_1{
magnitude_1/Abs
ComplexAbsstft_1/transpose_1:y:0*1
_output_shapes
:         ╖Б2
magnitude_1/Absq
IdentityIdentitymagnitude_1/Abs:y:0*
T0*1
_output_shapes
:         ╖Б2

Identity"
identityIdentity:output:0*,
_input_shapes
:         Ат	:U Q
-
_output_shapes
:         Ат	
 
_user_specified_nameinputs
Э
и
5__inference_batch_normalization_7_layer_call_fn_46558

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCall┤
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Y
fTRR
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_442032
StatefulPartitionedCallи
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                            ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
ю	
█
B__inference_dense_5_layer_call_and_return_conditional_losses_44765

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:         2	
SigmoidР
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*.
_input_shapes
:         ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
Жи
Ю
G__inference_sequential_1_layer_call_and_return_conditional_losses_45456

inputs1
-batch_normalization_4_readvariableop_resource3
/batch_normalization_4_readvariableop_1_resourceB
>batch_normalization_4_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource+
'conv2d_3_conv2d_readvariableop_resource,
(conv2d_3_biasadd_readvariableop_resource1
-batch_normalization_5_readvariableop_resource3
/batch_normalization_5_readvariableop_1_resourceB
>batch_normalization_5_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource+
'conv2d_4_conv2d_readvariableop_resource,
(conv2d_4_biasadd_readvariableop_resource1
-batch_normalization_6_readvariableop_resource3
/batch_normalization_6_readvariableop_1_resourceB
>batch_normalization_6_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource+
'conv2d_5_conv2d_readvariableop_resource,
(conv2d_5_biasadd_readvariableop_resource1
-batch_normalization_7_readvariableop_resource3
/batch_normalization_7_readvariableop_1_resourceB
>batch_normalization_7_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
&dense_3_matmul_readvariableop_resource+
'dense_3_biasadd_readvariableop_resource*
&dense_4_matmul_readvariableop_resource+
'dense_4_biasadd_readvariableop_resource*
&dense_5_matmul_readvariableop_resource+
'dense_5_biasadd_readvariableop_resource
identityИв$batch_normalization_4/AssignNewValueв&batch_normalization_4/AssignNewValue_1в5batch_normalization_4/FusedBatchNormV3/ReadVariableOpв7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1в$batch_normalization_4/ReadVariableOpв&batch_normalization_4/ReadVariableOp_1в$batch_normalization_5/AssignNewValueв&batch_normalization_5/AssignNewValue_1в5batch_normalization_5/FusedBatchNormV3/ReadVariableOpв7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1в$batch_normalization_5/ReadVariableOpв&batch_normalization_5/ReadVariableOp_1в$batch_normalization_6/AssignNewValueв&batch_normalization_6/AssignNewValue_1в5batch_normalization_6/FusedBatchNormV3/ReadVariableOpв7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1в$batch_normalization_6/ReadVariableOpв&batch_normalization_6/ReadVariableOp_1в$batch_normalization_7/AssignNewValueв&batch_normalization_7/AssignNewValue_1в5batch_normalization_7/FusedBatchNormV3/ReadVariableOpв7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1в$batch_normalization_7/ReadVariableOpв&batch_normalization_7/ReadVariableOp_1вconv2d_3/BiasAdd/ReadVariableOpвconv2d_3/Conv2D/ReadVariableOpвconv2d_4/BiasAdd/ReadVariableOpвconv2d_4/Conv2D/ReadVariableOpвconv2d_5/BiasAdd/ReadVariableOpвconv2d_5/Conv2D/ReadVariableOpвdense_3/BiasAdd/ReadVariableOpвdense_3/MatMul/ReadVariableOpвdense_4/BiasAdd/ReadVariableOpвdense_4/MatMul/ReadVariableOpвdense_5/BiasAdd/ReadVariableOpвdense_5/MatMul/ReadVariableOpX
reshape_1/ShapeShapeinputs*
T0*
_output_shapes
:2
reshape_1/ShapeИ
reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_1/strided_slice/stackМ
reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_1/strided_slice/stack_1М
reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_1/strided_slice/stack_2Ю
reshape_1/strided_sliceStridedSlicereshape_1/Shape:output:0&reshape_1/strided_slice/stack:output:0(reshape_1/strided_slice/stack_1:output:0(reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_1/strided_slicez
reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB	 :Ат	2
reshape_1/Reshape/shape/1x
reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_1/Reshape/shape/2╥
reshape_1/Reshape/shapePack reshape_1/strided_slice:output:0"reshape_1/Reshape/shape/1:output:0"reshape_1/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape_1/Reshape/shapeУ
reshape_1/ReshapeReshapeinputs reshape_1/Reshape/shape:output:0*
T0*-
_output_shapes
:         Ат	2
reshape_1/Reshapeб
$stft_magnitude/stft_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2&
$stft_magnitude/stft_1/transpose/perm╥
stft_magnitude/stft_1/transpose	Transposereshape_1/Reshape:output:0-stft_magnitude/stft_1/transpose/perm:output:0*
T0*-
_output_shapes
:         Ат	2!
stft_magnitude/stft_1/transpose╖
8stft_magnitude/stft_1/stft_1_tf.signal.stft/frame_lengthConst*
_output_shapes
: *
dtype0*
value
B :А2:
8stft_magnitude/stft_1/stft_1_tf.signal.stft/frame_length│
6stft_magnitude/stft_1/stft_1_tf.signal.stft/frame_stepConst*
_output_shapes
: *
dtype0*
value
B :А28
6stft_magnitude/stft_1/stft_1_tf.signal.stft/frame_step│
6stft_magnitude/stft_1/stft_1_tf.signal.stft/fft_lengthConst*
_output_shapes
: *
dtype0*
value
B :А28
6stft_magnitude/stft_1/stft_1_tf.signal.stft/fft_length╗
6stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/axisConst*
_output_shapes
: *
dtype0*
valueB :
         28
6stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/axis┼
7stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/ShapeShape#stft_magnitude/stft_1/transpose:y:0*
T0*
_output_shapes
:29
7stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/Shape▓
6stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/RankConst*
_output_shapes
: *
dtype0*
value	B :28
6stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/Rank└
=stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2?
=stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/range/start└
=stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2?
=stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/range/deltaш
7stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/rangeRangeFstft_magnitude/stft_1/stft_1_tf.signal.stft/frame/range/start:output:0?stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/Rank:output:0Fstft_magnitude/stft_1/stft_1_tf.signal.stft/frame/range/delta:output:0*
_output_shapes
:29
7stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/rangeс
Estft_magnitude/stft_1/stft_1_tf.signal.stft/frame/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2G
Estft_magnitude/stft_1/stft_1_tf.signal.stft/frame/strided_slice/stack▄
Gstft_magnitude/stft_1/stft_1_tf.signal.stft/frame/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2I
Gstft_magnitude/stft_1/stft_1_tf.signal.stft/frame/strided_slice/stack_1▄
Gstft_magnitude/stft_1/stft_1_tf.signal.stft/frame/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2I
Gstft_magnitude/stft_1/stft_1_tf.signal.stft/frame/strided_slice/stack_2О
?stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/strided_sliceStridedSlice@stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/range:output:0Nstft_magnitude/stft_1/stft_1_tf.signal.stft/frame/strided_slice/stack:output:0Pstft_magnitude/stft_1/stft_1_tf.signal.stft/frame/strided_slice/stack_1:output:0Pstft_magnitude/stft_1/stft_1_tf.signal.stft/frame/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2A
?stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/strided_slice┤
7stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/sub/yConst*
_output_shapes
: *
dtype0*
value	B :29
7stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/sub/yЩ
5stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/subSub?stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/Rank:output:0@stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/sub/y:output:0*
T0*
_output_shapes
: 27
5stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/subЯ
7stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/sub_1Sub9stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/sub:z:0Hstft_magnitude/stft_1/stft_1_tf.signal.stft/frame/strided_slice:output:0*
T0*
_output_shapes
: 29
7stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/sub_1║
:stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2<
:stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/packed/1Ў
8stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/packedPackHstft_magnitude/stft_1/stft_1_tf.signal.stft/frame/strided_slice:output:0Cstft_magnitude/stft_1/stft_1_tf.signal.stft/frame/packed/1:output:0;stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/sub_1:z:0*
N*
T0*
_output_shapes
:2:
8stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/packed╚
Astft_magnitude/stft_1/stft_1_tf.signal.stft/frame/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2C
Astft_magnitude/stft_1/stft_1_tf.signal.stft/frame/split/split_dimЩ
7stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/splitSplitV@stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/Shape:output:0Astft_magnitude/stft_1/stft_1_tf.signal.stft/frame/packed:output:0Jstft_magnitude/stft_1/stft_1_tf.signal.stft/frame/split/split_dim:output:0*
T0*

Tlen0*$
_output_shapes
::: *
	num_split29
7stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/split┼
?stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/Reshape/shapeConst*
_output_shapes
: *
dtype0*
valueB 2A
?stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/Reshape/shape╔
Astft_magnitude/stft_1/stft_1_tf.signal.stft/frame/Reshape/shape_1Const*
_output_shapes
: *
dtype0*
valueB 2C
Astft_magnitude/stft_1/stft_1_tf.signal.stft/frame/Reshape/shape_1░
9stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/ReshapeReshape@stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/split:output:1Jstft_magnitude/stft_1/stft_1_tf.signal.stft/frame/Reshape/shape_1:output:0*
T0*
_output_shapes
: 2;
9stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/Reshape▓
6stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/SizeConst*
_output_shapes
: *
dtype0*
value	B :28
6stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/Size╢
8stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/Size_1Const*
_output_shapes
: *
dtype0*
value	B : 2:
8stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/Size_1б
7stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/sub_2SubBstft_magnitude/stft_1/stft_1_tf.signal.stft/frame/Reshape:output:0Astft_magnitude/stft_1/stft_1_tf.signal.stft/frame_length:output:0*
T0*
_output_shapes
: 29
7stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/sub_2г
:stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/floordivFloorDiv;stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/sub_2:z:0?stft_magnitude/stft_1/stft_1_tf.signal.stft/frame_step:output:0*
T0*
_output_shapes
: 2<
:stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/floordiv┤
7stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/add/xConst*
_output_shapes
: *
dtype0*
value	B :29
7stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/add/xЪ
5stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/addAddV2@stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/add/x:output:0>stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/floordiv:z:0*
T0*
_output_shapes
: 27
5stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/add╝
;stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/Maximum/xConst*
_output_shapes
: *
dtype0*
value	B : 2=
;stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/Maximum/xг
9stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/MaximumMaximumDstft_magnitude/stft_1/stft_1_tf.signal.stft/frame/Maximum/x:output:09stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/add:z:0*
T0*
_output_shapes
: 2;
9stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/Maximum╜
;stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/gcd/ConstConst*
_output_shapes
: *
dtype0*
value
B :А2=
;stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/gcd/Const├
>stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/floordiv_1/yConst*
_output_shapes
: *
dtype0*
value
B :А2@
>stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/floordiv_1/y╡
<stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/floordiv_1FloorDivAstft_magnitude/stft_1/stft_1_tf.signal.stft/frame_length:output:0Gstft_magnitude/stft_1/stft_1_tf.signal.stft/frame/floordiv_1/y:output:0*
T0*
_output_shapes
: 2>
<stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/floordiv_1├
>stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/floordiv_2/yConst*
_output_shapes
: *
dtype0*
value
B :А2@
>stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/floordiv_2/y│
<stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/floordiv_2FloorDiv?stft_magnitude/stft_1/stft_1_tf.signal.stft/frame_step:output:0Gstft_magnitude/stft_1/stft_1_tf.signal.stft/frame/floordiv_2/y:output:0*
T0*
_output_shapes
: 2>
<stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/floordiv_2├
>stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/floordiv_3/yConst*
_output_shapes
: *
dtype0*
value
B :А2@
>stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/floordiv_3/y╢
<stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/floordiv_3FloorDivBstft_magnitude/stft_1/stft_1_tf.signal.stft/frame/Reshape:output:0Gstft_magnitude/stft_1/stft_1_tf.signal.stft/frame/floordiv_3/y:output:0*
T0*
_output_shapes
: 2>
<stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/floordiv_3╡
7stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/mul/yConst*
_output_shapes
: *
dtype0*
value
B :А29
7stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/mul/yЪ
5stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/mulMul@stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/floordiv_3:z:0@stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/mul/y:output:0*
T0*
_output_shapes
: 27
5stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/mulў
Astft_magnitude/stft_1/stft_1_tf.signal.stft/frame/concat/values_1Pack9stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/mul:z:0*
N*
T0*
_output_shapes
:2C
Astft_magnitude/stft_1/stft_1_tf.signal.stft/frame/concat/values_1└
=stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2?
=stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/concat/axis╞
8stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/concatConcatV2@stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/split:output:0Jstft_magnitude/stft_1/stft_1_tf.signal.stft/frame/concat/values_1:output:0@stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/split:output:2Fstft_magnitude/stft_1/stft_1_tf.signal.stft/frame/concat/axis:output:0*
N*
T0*
_output_shapes
:2:
8stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/concat╤
Estft_magnitude/stft_1/stft_1_tf.signal.stft/frame/concat_1/values_1/1Const*
_output_shapes
: *
dtype0*
value
B :А2G
Estft_magnitude/stft_1/stft_1_tf.signal.stft/frame/concat_1/values_1/1╥
Cstft_magnitude/stft_1/stft_1_tf.signal.stft/frame/concat_1/values_1Pack@stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/floordiv_3:z:0Nstft_magnitude/stft_1/stft_1_tf.signal.stft/frame/concat_1/values_1/1:output:0*
N*
T0*
_output_shapes
:2E
Cstft_magnitude/stft_1/stft_1_tf.signal.stft/frame/concat_1/values_1─
?stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2A
?stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/concat_1/axis╬
:stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/concat_1ConcatV2@stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/split:output:0Lstft_magnitude/stft_1/stft_1_tf.signal.stft/frame/concat_1/values_1:output:0@stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/split:output:2Hstft_magnitude/stft_1/stft_1_tf.signal.stft/frame/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2<
:stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/concat_1╞
<stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/zeros_likeConst*
_output_shapes
:*
dtype0*
valueB: 2>
<stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/zeros_like╨
Astft_magnitude/stft_1/stft_1_tf.signal.stft/frame/ones_like/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2C
Astft_magnitude/stft_1/stft_1_tf.signal.stft/frame/ones_like/Shape╚
Astft_magnitude/stft_1/stft_1_tf.signal.stft/frame/ones_like/ConstConst*
_output_shapes
: *
dtype0*
value	B :2C
Astft_magnitude/stft_1/stft_1_tf.signal.stft/frame/ones_like/Const┐
;stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/ones_likeFillJstft_magnitude/stft_1/stft_1_tf.signal.stft/frame/ones_like/Shape:output:0Jstft_magnitude/stft_1/stft_1_tf.signal.stft/frame/ones_like/Const:output:0*
T0*
_output_shapes
:2=
;stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/ones_like┌
>stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/StridedSliceStridedSlice#stft_magnitude/stft_1/transpose:y:0Estft_magnitude/stft_1/stft_1_tf.signal.stft/frame/zeros_like:output:0Astft_magnitude/stft_1/stft_1_tf.signal.stft/frame/concat:output:0Dstft_magnitude/stft_1/stft_1_tf.signal.stft/frame/ones_like:output:0*
Index0*
T0*=
_output_shapes+
):'                           2@
>stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/StridedSliceр
;stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/Reshape_1ReshapeGstft_magnitude/stft_1/stft_1_tf.signal.stft/frame/StridedSlice:output:0Cstft_magnitude/stft_1/stft_1_tf.signal.stft/frame/concat_1:output:0*
T0*B
_output_shapes0
.:,                           А2=
;stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/Reshape_1─
?stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/range_1/startConst*
_output_shapes
: *
dtype0*
value	B : 2A
?stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/range_1/start─
?stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :2A
?stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/range_1/deltaў
9stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/range_1RangeHstft_magnitude/stft_1/stft_1_tf.signal.stft/frame/range_1/start:output:0=stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/Maximum:z:0Hstft_magnitude/stft_1/stft_1_tf.signal.stft/frame/range_1/delta:output:0*#
_output_shapes
:         2;
9stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/range_1н
7stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/mul_1MulBstft_magnitude/stft_1/stft_1_tf.signal.stft/frame/range_1:output:0@stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/floordiv_2:z:0*
T0*#
_output_shapes
:         29
7stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/mul_1╠
Cstft_magnitude/stft_1/stft_1_tf.signal.stft/frame/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2E
Cstft_magnitude/stft_1/stft_1_tf.signal.stft/frame/Reshape_2/shape/1╔
Astft_magnitude/stft_1/stft_1_tf.signal.stft/frame/Reshape_2/shapePack=stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/Maximum:z:0Lstft_magnitude/stft_1/stft_1_tf.signal.stft/frame/Reshape_2/shape/1:output:0*
N*
T0*
_output_shapes
:2C
Astft_magnitude/stft_1/stft_1_tf.signal.stft/frame/Reshape_2/shape└
;stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/Reshape_2Reshape;stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/mul_1:z:0Jstft_magnitude/stft_1/stft_1_tf.signal.stft/frame/Reshape_2/shape:output:0*
T0*'
_output_shapes
:         2=
;stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/Reshape_2─
?stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/range_2/startConst*
_output_shapes
: *
dtype0*
value	B : 2A
?stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/range_2/start─
?stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/range_2/deltaConst*
_output_shapes
: *
dtype0*
value	B :2A
?stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/range_2/deltaё
9stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/range_2RangeHstft_magnitude/stft_1/stft_1_tf.signal.stft/frame/range_2/start:output:0@stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/floordiv_1:z:0Hstft_magnitude/stft_1/stft_1_tf.signal.stft/frame/range_2/delta:output:0*
_output_shapes
:2;
9stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/range_2╠
Cstft_magnitude/stft_1/stft_1_tf.signal.stft/frame/Reshape_3/shape/0Const*
_output_shapes
: *
dtype0*
value	B :2E
Cstft_magnitude/stft_1/stft_1_tf.signal.stft/frame/Reshape_3/shape/0╠
Astft_magnitude/stft_1/stft_1_tf.signal.stft/frame/Reshape_3/shapePackLstft_magnitude/stft_1/stft_1_tf.signal.stft/frame/Reshape_3/shape/0:output:0@stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/floordiv_1:z:0*
N*
T0*
_output_shapes
:2C
Astft_magnitude/stft_1/stft_1_tf.signal.stft/frame/Reshape_3/shape╛
;stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/Reshape_3ReshapeBstft_magnitude/stft_1/stft_1_tf.signal.stft/frame/range_2:output:0Jstft_magnitude/stft_1/stft_1_tf.signal.stft/frame/Reshape_3/shape:output:0*
T0*
_output_shapes

:2=
;stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/Reshape_3╣
7stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/add_1AddV2Dstft_magnitude/stft_1/stft_1_tf.signal.stft/frame/Reshape_2:output:0Dstft_magnitude/stft_1/stft_1_tf.signal.stft/frame/Reshape_3:output:0*
T0*'
_output_shapes
:         29
7stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/add_1┼
:stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/GatherV2GatherV2Dstft_magnitude/stft_1/stft_1_tf.signal.stft/frame/Reshape_1:output:0;stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/add_1:z:0Hstft_magnitude/stft_1/stft_1_tf.signal.stft/frame/strided_slice:output:0*
Taxis0*
Tindices0*
Tparams0*F
_output_shapes4
2:0                           А2<
:stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/GatherV2┬
Cstft_magnitude/stft_1/stft_1_tf.signal.stft/frame/concat_2/values_1Pack=stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/Maximum:z:0Astft_magnitude/stft_1/stft_1_tf.signal.stft/frame_length:output:0*
N*
T0*
_output_shapes
:2E
Cstft_magnitude/stft_1/stft_1_tf.signal.stft/frame/concat_2/values_1─
?stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2A
?stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/concat_2/axis╬
:stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/concat_2ConcatV2@stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/split:output:0Lstft_magnitude/stft_1/stft_1_tf.signal.stft/frame/concat_2/values_1:output:0@stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/split:output:2Hstft_magnitude/stft_1/stft_1_tf.signal.stft/frame/concat_2/axis:output:0*
N*
T0*
_output_shapes
:2<
:stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/concat_2╦
;stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/Reshape_4ReshapeCstft_magnitude/stft_1/stft_1_tf.signal.stft/frame/GatherV2:output:0Cstft_magnitude/stft_1/stft_1_tf.signal.stft/frame/concat_2:output:0*
T0*1
_output_shapes
:         ╖А2=
;stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/Reshape_4╞
@stft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/periodicConst*
_output_shapes
: *
dtype0
*
value	B
 Z2B
@stft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/periodic 
<stft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/CastCastIstft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/periodic:output:0*

DstT0*

SrcT0
*
_output_shapes
: 2>
<stft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/Cast╩
Bstft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/FloorMod/yConst*
_output_shapes
: *
dtype0*
value	B :2D
Bstft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/FloorMod/y┴
@stft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/FloorModFloorModAstft_magnitude/stft_1/stft_1_tf.signal.stft/frame_length:output:0Kstft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/FloorMod/y:output:0*
T0*
_output_shapes
: 2B
@stft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/FloorMod└
=stft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/sub/xConst*
_output_shapes
: *
dtype0*
value	B :2?
=stft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/sub/x░
;stft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/subSubFstft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/sub/x:output:0Dstft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/FloorMod:z:0*
T0*
_output_shapes
: 2=
;stft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/subе
;stft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/mulMul@stft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/Cast:y:0?stft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/sub:z:0*
T0*
_output_shapes
: 2=
;stft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/mulи
;stft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/addAddV2Astft_magnitude/stft_1/stft_1_tf.signal.stft/frame_length:output:0?stft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/mul:z:0*
T0*
_output_shapes
: 2=
;stft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/add─
?stft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/sub_1/yConst*
_output_shapes
: *
dtype0*
value	B :2A
?stft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/sub_1/y▒
=stft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/sub_1Sub?stft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/add:z:0Hstft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/sub_1/y:output:0*
T0*
_output_shapes
: 2?
=stft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/sub_1√
>stft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/Cast_1CastAstft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/sub_1:z:0*

DstT0*

SrcT0*
_output_shapes
: 2@
>stft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/Cast_1╠
Cstft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2E
Cstft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/range/start╠
Cstft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2E
Cstft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/range/deltaГ
=stft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/rangeRangeLstft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/range/start:output:0Astft_magnitude/stft_1/stft_1_tf.signal.stft/frame_length:output:0Lstft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/range/delta:output:0*
_output_shapes	
:А2?
=stft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/rangeЕ
>stft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/Cast_2CastFstft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/range:output:0*

DstT0*

SrcT0*
_output_shapes	
:А2@
>stft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/Cast_2├
=stft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *█╔@2?
=stft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/Const╖
=stft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/mul_1MulFstft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/Const:output:0Bstft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/Cast_2:y:0*
T0*
_output_shapes	
:А2?
=stft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/mul_1║
?stft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/truedivRealDivAstft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/mul_1:z:0Bstft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/Cast_1:y:0*
T0*
_output_shapes	
:А2A
?stft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/truedivь
;stft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/CosCosCstft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/truediv:z:0*
T0*
_output_shapes	
:А2=
;stft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/Cos╟
?stft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2A
?stft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/mul_2/x╢
=stft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/mul_2MulHstft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/mul_2/x:output:0?stft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/Cos:y:0*
T0*
_output_shapes	
:А2?
=stft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/mul_2╟
?stft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2A
?stft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/sub_2/x╕
=stft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/sub_2SubHstft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/sub_2/x:output:0Astft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/mul_2:z:0*
T0*
_output_shapes	
:А2?
=stft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/sub_2о
/stft_magnitude/stft_1/stft_1_tf.signal.stft/mulMulDstft_magnitude/stft_1/stft_1_tf.signal.stft/frame/Reshape_4:output:0Astft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/sub_2:z:0*
T0*1
_output_shapes
:         ╖А21
/stft_magnitude/stft_1/stft_1_tf.signal.stft/mulщ
7stft_magnitude/stft_1/stft_1_tf.signal.stft/rfft/packedPack?stft_magnitude/stft_1/stft_1_tf.signal.stft/fft_length:output:0*
N*
T0*
_output_shapes
:29
7stft_magnitude/stft_1/stft_1_tf.signal.stft/rfft/packed┼
;stft_magnitude/stft_1/stft_1_tf.signal.stft/rfft/fft_lengthConst*
_output_shapes
:*
dtype0*
valueB:А2=
;stft_magnitude/stft_1/stft_1_tf.signal.stft/rfft/fft_lengthЪ
0stft_magnitude/stft_1/stft_1_tf.signal.stft/rfftRFFT3stft_magnitude/stft_1/stft_1_tf.signal.stft/mul:z:0Dstft_magnitude/stft_1/stft_1_tf.signal.stft/rfft/fft_length:output:0*1
_output_shapes
:         ╖Б22
0stft_magnitude/stft_1/stft_1_tf.signal.stft/rfftй
&stft_magnitude/stft_1/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             2(
&stft_magnitude/stft_1/transpose_1/perm√
!stft_magnitude/stft_1/transpose_1	Transpose9stft_magnitude/stft_1/stft_1_tf.signal.stft/rfft:output:0/stft_magnitude/stft_1/transpose_1/perm:output:0*
T0*1
_output_shapes
:         ╖Б2#
!stft_magnitude/stft_1/transpose_1и
stft_magnitude/magnitude_1/Abs
ComplexAbs%stft_magnitude/stft_1/transpose_1:y:0*1
_output_shapes
:         ╖Б2 
stft_magnitude/magnitude_1/Abs╢
$batch_normalization_4/ReadVariableOpReadVariableOp-batch_normalization_4_readvariableop_resource*
_output_shapes
:*
dtype02&
$batch_normalization_4/ReadVariableOp╝
&batch_normalization_4/ReadVariableOp_1ReadVariableOp/batch_normalization_4_readvariableop_1_resource*
_output_shapes
:*
dtype02(
&batch_normalization_4/ReadVariableOp_1щ
5batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_4/FusedBatchNormV3/ReadVariableOpя
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype029
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1·
&batch_normalization_4/FusedBatchNormV3FusedBatchNormV3"stft_magnitude/magnitude_1/Abs:y:0,batch_normalization_4/ReadVariableOp:value:0.batch_normalization_4/ReadVariableOp_1:value:0=batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:         ╖Б:::::*
epsilon%oГ:*
exponential_avg_factor%
╫#<2(
&batch_normalization_4/FusedBatchNormV3▒
$batch_normalization_4/AssignNewValueAssignVariableOp>batch_normalization_4_fusedbatchnormv3_readvariableop_resource3batch_normalization_4/FusedBatchNormV3:batch_mean:06^batch_normalization_4/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*Q
_classG
ECloc:@batch_normalization_4/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02&
$batch_normalization_4/AssignNewValue┐
&batch_normalization_4/AssignNewValue_1AssignVariableOp@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_4/FusedBatchNormV3:batch_variance:08^batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*S
_classI
GEloc:@batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02(
&batch_normalization_4/AssignNewValue_1░
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02 
conv2d_3/Conv2D/ReadVariableOpф
conv2d_3/Conv2DConv2D*batch_normalization_4/FusedBatchNormV3:y:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ╖Б*
paddingSAME*
strides
2
conv2d_3/Conv2Dз
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_3/BiasAdd/ReadVariableOpо
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ╖Б2
conv2d_3/BiasAdd}
conv2d_3/ReluReluconv2d_3/BiasAdd:output:0*
T0*1
_output_shapes
:         ╖Б2
conv2d_3/Relu╚
max_pooling2d_3/MaxPoolMaxPoolconv2d_3/Relu:activations:0*0
_output_shapes
:         gл*
ksize
*
paddingVALID*
strides
2
max_pooling2d_3/MaxPool╢
$batch_normalization_5/ReadVariableOpReadVariableOp-batch_normalization_5_readvariableop_resource*
_output_shapes
:*
dtype02&
$batch_normalization_5/ReadVariableOp╝
&batch_normalization_5/ReadVariableOp_1ReadVariableOp/batch_normalization_5_readvariableop_1_resource*
_output_shapes
:*
dtype02(
&batch_normalization_5/ReadVariableOp_1щ
5batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_5/FusedBatchNormV3/ReadVariableOpя
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype029
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ў
&batch_normalization_5/FusedBatchNormV3FusedBatchNormV3 max_pooling2d_3/MaxPool:output:0,batch_normalization_5/ReadVariableOp:value:0.batch_normalization_5/ReadVariableOp_1:value:0=batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:         gл:::::*
epsilon%oГ:*
exponential_avg_factor%
╫#<2(
&batch_normalization_5/FusedBatchNormV3▒
$batch_normalization_5/AssignNewValueAssignVariableOp>batch_normalization_5_fusedbatchnormv3_readvariableop_resource3batch_normalization_5/FusedBatchNormV3:batch_mean:06^batch_normalization_5/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*Q
_classG
ECloc:@batch_normalization_5/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02&
$batch_normalization_5/AssignNewValue┐
&batch_normalization_5/AssignNewValue_1AssignVariableOp@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_5/FusedBatchNormV3:batch_variance:08^batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*S
_classI
GEloc:@batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02(
&batch_normalization_5/AssignNewValue_1░
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02 
conv2d_4/Conv2D/ReadVariableOpу
conv2d_4/Conv2DConv2D*batch_normalization_5/FusedBatchNormV3:y:0&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         gл*
paddingSAME*
strides
2
conv2d_4/Conv2Dз
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_4/BiasAdd/ReadVariableOpн
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         gл2
conv2d_4/BiasAdd|
conv2d_4/ReluReluconv2d_4/BiasAdd:output:0*
T0*0
_output_shapes
:         gл2
conv2d_4/Relu╟
max_pooling2d_4/MaxPoolMaxPoolconv2d_4/Relu:activations:0*/
_output_shapes
:         39*
ksize
*
paddingVALID*
strides
2
max_pooling2d_4/MaxPool╢
$batch_normalization_6/ReadVariableOpReadVariableOp-batch_normalization_6_readvariableop_resource*
_output_shapes
:*
dtype02&
$batch_normalization_6/ReadVariableOp╝
&batch_normalization_6/ReadVariableOp_1ReadVariableOp/batch_normalization_6_readvariableop_1_resource*
_output_shapes
:*
dtype02(
&batch_normalization_6/ReadVariableOp_1щ
5batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_6/FusedBatchNormV3/ReadVariableOpя
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype029
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1Ў
&batch_normalization_6/FusedBatchNormV3FusedBatchNormV3 max_pooling2d_4/MaxPool:output:0,batch_normalization_6/ReadVariableOp:value:0.batch_normalization_6/ReadVariableOp_1:value:0=batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         39:::::*
epsilon%oГ:*
exponential_avg_factor%
╫#<2(
&batch_normalization_6/FusedBatchNormV3▒
$batch_normalization_6/AssignNewValueAssignVariableOp>batch_normalization_6_fusedbatchnormv3_readvariableop_resource3batch_normalization_6/FusedBatchNormV3:batch_mean:06^batch_normalization_6/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*Q
_classG
ECloc:@batch_normalization_6/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02&
$batch_normalization_6/AssignNewValue┐
&batch_normalization_6/AssignNewValue_1AssignVariableOp@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_6/FusedBatchNormV3:batch_variance:08^batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*S
_classI
GEloc:@batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02(
&batch_normalization_6/AssignNewValue_1░
conv2d_5/Conv2D/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02 
conv2d_5/Conv2D/ReadVariableOpт
conv2d_5/Conv2DConv2D*batch_normalization_6/FusedBatchNormV3:y:0&conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         39 *
paddingSAME*
strides
2
conv2d_5/Conv2Dз
conv2d_5/BiasAdd/ReadVariableOpReadVariableOp(conv2d_5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_5/BiasAdd/ReadVariableOpм
conv2d_5/BiasAddBiasAddconv2d_5/Conv2D:output:0'conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         39 2
conv2d_5/BiasAdd{
conv2d_5/ReluReluconv2d_5/BiasAdd:output:0*
T0*/
_output_shapes
:         39 2
conv2d_5/Relu╟
max_pooling2d_5/MaxPoolMaxPoolconv2d_5/Relu:activations:0*/
_output_shapes
:          *
ksize
*
paddingVALID*
strides
2
max_pooling2d_5/MaxPool╢
$batch_normalization_7/ReadVariableOpReadVariableOp-batch_normalization_7_readvariableop_resource*
_output_shapes
: *
dtype02&
$batch_normalization_7/ReadVariableOp╝
&batch_normalization_7/ReadVariableOp_1ReadVariableOp/batch_normalization_7_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&batch_normalization_7/ReadVariableOp_1щ
5batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype027
5batch_normalization_7/FusedBatchNormV3/ReadVariableOpя
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype029
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1Ў
&batch_normalization_7/FusedBatchNormV3FusedBatchNormV3 max_pooling2d_5/MaxPool:output:0,batch_normalization_7/ReadVariableOp:value:0.batch_normalization_7/ReadVariableOp_1:value:0=batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:          : : : : :*
epsilon%oГ:*
exponential_avg_factor%
╫#<2(
&batch_normalization_7/FusedBatchNormV3▒
$batch_normalization_7/AssignNewValueAssignVariableOp>batch_normalization_7_fusedbatchnormv3_readvariableop_resource3batch_normalization_7/FusedBatchNormV3:batch_mean:06^batch_normalization_7/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*Q
_classG
ECloc:@batch_normalization_7/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02&
$batch_normalization_7/AssignNewValue┐
&batch_normalization_7/AssignNewValue_1AssignVariableOp@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_7/FusedBatchNormV3:batch_variance:08^batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*S
_classI
GEloc:@batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02(
&batch_normalization_7/AssignNewValue_1s
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"    `;  2
flatten_1/Constк
flatten_1/ReshapeReshape*batch_normalization_7/FusedBatchNormV3:y:0flatten_1/Const:output:0*
T0*(
_output_shapes
:         рv2
flatten_1/Reshapeж
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes
:	рv@*
dtype02
dense_3/MatMul/ReadVariableOpЯ
dense_3/MatMulMatMulflatten_1/Reshape:output:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2
dense_3/MatMulд
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
dense_3/BiasAdd/ReadVariableOpб
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2
dense_3/BiasAddp
dense_3/ReluReludense_3/BiasAdd:output:0*
T0*'
_output_shapes
:         @2
dense_3/Reluw
dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ЧЦЦ?2
dropout_2/dropout/Constе
dropout_2/dropout/MulMuldense_3/Relu:activations:0 dropout_2/dropout/Const:output:0*
T0*'
_output_shapes
:         @2
dropout_2/dropout/Mul|
dropout_2/dropout/ShapeShapedense_3/Relu:activations:0*
T0*
_output_shapes
:2
dropout_2/dropout/Shape╥
.dropout_2/dropout/random_uniform/RandomUniformRandomUniform dropout_2/dropout/Shape:output:0*
T0*'
_output_shapes
:         @*
dtype020
.dropout_2/dropout/random_uniform/RandomUniformЙ
 dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩ>2"
 dropout_2/dropout/GreaterEqual/yц
dropout_2/dropout/GreaterEqualGreaterEqual7dropout_2/dropout/random_uniform/RandomUniform:output:0)dropout_2/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         @2 
dropout_2/dropout/GreaterEqualЭ
dropout_2/dropout/CastCast"dropout_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         @2
dropout_2/dropout/Castв
dropout_2/dropout/Mul_1Muldropout_2/dropout/Mul:z:0dropout_2/dropout/Cast:y:0*
T0*'
_output_shapes
:         @2
dropout_2/dropout/Mul_1е
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02
dense_4/MatMul/ReadVariableOpа
dense_4/MatMulMatMuldropout_2/dropout/Mul_1:z:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_4/MatMulд
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_4/BiasAdd/ReadVariableOpб
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_4/BiasAddp
dense_4/ReluReludense_4/BiasAdd:output:0*
T0*'
_output_shapes
:         2
dense_4/Reluw
dropout_3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *]tС?2
dropout_3/dropout/Constе
dropout_3/dropout/MulMuldense_4/Relu:activations:0 dropout_3/dropout/Const:output:0*
T0*'
_output_shapes
:         2
dropout_3/dropout/Mul|
dropout_3/dropout/ShapeShapedense_4/Relu:activations:0*
T0*
_output_shapes
:2
dropout_3/dropout/Shape╥
.dropout_3/dropout/random_uniform/RandomUniformRandomUniform dropout_3/dropout/Shape:output:0*
T0*'
_output_shapes
:         *
dtype020
.dropout_3/dropout/random_uniform/RandomUniformЙ
 dropout_3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *П┬ї=2"
 dropout_3/dropout/GreaterEqual/yц
dropout_3/dropout/GreaterEqualGreaterEqual7dropout_3/dropout/random_uniform/RandomUniform:output:0)dropout_3/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         2 
dropout_3/dropout/GreaterEqualЭ
dropout_3/dropout/CastCast"dropout_3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         2
dropout_3/dropout/Castв
dropout_3/dropout/Mul_1Muldropout_3/dropout/Mul:z:0dropout_3/dropout/Cast:y:0*
T0*'
_output_shapes
:         2
dropout_3/dropout/Mul_1е
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_5/MatMul/ReadVariableOpа
dense_5/MatMulMatMuldropout_3/dropout/Mul_1:z:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_5/MatMulд
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_5/BiasAdd/ReadVariableOpб
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_5/BiasAddy
dense_5/SigmoidSigmoiddense_5/BiasAdd:output:0*
T0*'
_output_shapes
:         2
dense_5/Sigmoid╗
IdentityIdentitydense_5/Sigmoid:y:0%^batch_normalization_4/AssignNewValue'^batch_normalization_4/AssignNewValue_16^batch_normalization_4/FusedBatchNormV3/ReadVariableOp8^batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_4/ReadVariableOp'^batch_normalization_4/ReadVariableOp_1%^batch_normalization_5/AssignNewValue'^batch_normalization_5/AssignNewValue_16^batch_normalization_5/FusedBatchNormV3/ReadVariableOp8^batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_5/ReadVariableOp'^batch_normalization_5/ReadVariableOp_1%^batch_normalization_6/AssignNewValue'^batch_normalization_6/AssignNewValue_16^batch_normalization_6/FusedBatchNormV3/ReadVariableOp8^batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_6/ReadVariableOp'^batch_normalization_6/ReadVariableOp_1%^batch_normalization_7/AssignNewValue'^batch_normalization_7/AssignNewValue_16^batch_normalization_7/FusedBatchNormV3/ReadVariableOp8^batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_7/ReadVariableOp'^batch_normalization_7/ReadVariableOp_1 ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp ^conv2d_4/BiasAdd/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp ^conv2d_5/BiasAdd/ReadVariableOp^conv2d_5/Conv2D/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*Ъ
_input_shapesИ
Е:         Ат	::::::::::::::::::::::::::::2L
$batch_normalization_4/AssignNewValue$batch_normalization_4/AssignNewValue2P
&batch_normalization_4/AssignNewValue_1&batch_normalization_4/AssignNewValue_12n
5batch_normalization_4/FusedBatchNormV3/ReadVariableOp5batch_normalization_4/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_17batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_4/ReadVariableOp$batch_normalization_4/ReadVariableOp2P
&batch_normalization_4/ReadVariableOp_1&batch_normalization_4/ReadVariableOp_12L
$batch_normalization_5/AssignNewValue$batch_normalization_5/AssignNewValue2P
&batch_normalization_5/AssignNewValue_1&batch_normalization_5/AssignNewValue_12n
5batch_normalization_5/FusedBatchNormV3/ReadVariableOp5batch_normalization_5/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_17batch_normalization_5/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_5/ReadVariableOp$batch_normalization_5/ReadVariableOp2P
&batch_normalization_5/ReadVariableOp_1&batch_normalization_5/ReadVariableOp_12L
$batch_normalization_6/AssignNewValue$batch_normalization_6/AssignNewValue2P
&batch_normalization_6/AssignNewValue_1&batch_normalization_6/AssignNewValue_12n
5batch_normalization_6/FusedBatchNormV3/ReadVariableOp5batch_normalization_6/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_17batch_normalization_6/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_6/ReadVariableOp$batch_normalization_6/ReadVariableOp2P
&batch_normalization_6/ReadVariableOp_1&batch_normalization_6/ReadVariableOp_12L
$batch_normalization_7/AssignNewValue$batch_normalization_7/AssignNewValue2P
&batch_normalization_7/AssignNewValue_1&batch_normalization_7/AssignNewValue_12n
5batch_normalization_7/FusedBatchNormV3/ReadVariableOp5batch_normalization_7/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_17batch_normalization_7/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_7/ReadVariableOp$batch_normalization_7/ReadVariableOp2P
&batch_normalization_7/ReadVariableOp_1&batch_normalization_7/ReadVariableOp_12B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2B
conv2d_4/BiasAdd/ReadVariableOpconv2d_4/BiasAdd/ReadVariableOp2@
conv2d_4/Conv2D/ReadVariableOpconv2d_4/Conv2D/ReadVariableOp2B
conv2d_5/BiasAdd/ReadVariableOpconv2d_5/BiasAdd/ReadVariableOp2@
conv2d_5/Conv2D/ReadVariableOpconv2d_5/Conv2D/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp:Q M
)
_output_shapes
:         Ат	
 
_user_specified_nameinputs
А
c
D__inference_dropout_3_layer_call_and_return_conditional_losses_44736

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *]tС?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:         2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape┤
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:         *
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *П┬ї=2
dropout/GreaterEqual/y╛
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:         2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
°
є
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_46448

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1╩
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         39:::::*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3┌
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:         392

Identity"
identityIdentity:output:0*>
_input_shapes-
+:         39::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:         39
 
_user_specified_nameinputs
В
}
(__inference_conv2d_3_layer_call_fn_46198

inputs
unknown
	unknown_0
identityИвStatefulPartitionedCall¤
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ╖Б*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_conv2d_3_layer_call_and_return_conditional_losses_443342
StatefulPartitionedCallШ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:         ╖Б2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:         ╖Б::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:         ╖Б
 
_user_specified_nameinputs
Э
и
5__inference_batch_normalization_6_layer_call_fn_46410

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCall┤
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Y
fTRR
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_440872
StatefulPartitionedCallи
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
ИЩ
╜"
__inference__traced_save_47125
file_prefix:
6savev2_batch_normalization_4_gamma_read_readvariableop9
5savev2_batch_normalization_4_beta_read_readvariableop@
<savev2_batch_normalization_4_moving_mean_read_readvariableopD
@savev2_batch_normalization_4_moving_variance_read_readvariableop.
*savev2_conv2d_3_kernel_read_readvariableop,
(savev2_conv2d_3_bias_read_readvariableop:
6savev2_batch_normalization_5_gamma_read_readvariableop9
5savev2_batch_normalization_5_beta_read_readvariableop@
<savev2_batch_normalization_5_moving_mean_read_readvariableopD
@savev2_batch_normalization_5_moving_variance_read_readvariableop.
*savev2_conv2d_4_kernel_read_readvariableop,
(savev2_conv2d_4_bias_read_readvariableop:
6savev2_batch_normalization_6_gamma_read_readvariableop9
5savev2_batch_normalization_6_beta_read_readvariableop@
<savev2_batch_normalization_6_moving_mean_read_readvariableopD
@savev2_batch_normalization_6_moving_variance_read_readvariableop.
*savev2_conv2d_5_kernel_read_readvariableop,
(savev2_conv2d_5_bias_read_readvariableop:
6savev2_batch_normalization_7_gamma_read_readvariableop9
5savev2_batch_normalization_7_beta_read_readvariableop@
<savev2_batch_normalization_7_moving_mean_read_readvariableopD
@savev2_batch_normalization_7_moving_variance_read_readvariableop-
)savev2_dense_3_kernel_read_readvariableop+
'savev2_dense_3_bias_read_readvariableop-
)savev2_dense_4_kernel_read_readvariableop+
'savev2_dense_4_bias_read_readvariableop-
)savev2_dense_5_kernel_read_readvariableop+
'savev2_dense_5_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableopA
=savev2_adam_batch_normalization_4_gamma_m_read_readvariableop@
<savev2_adam_batch_normalization_4_beta_m_read_readvariableop5
1savev2_adam_conv2d_3_kernel_m_read_readvariableop3
/savev2_adam_conv2d_3_bias_m_read_readvariableopA
=savev2_adam_batch_normalization_5_gamma_m_read_readvariableop@
<savev2_adam_batch_normalization_5_beta_m_read_readvariableop5
1savev2_adam_conv2d_4_kernel_m_read_readvariableop3
/savev2_adam_conv2d_4_bias_m_read_readvariableopA
=savev2_adam_batch_normalization_6_gamma_m_read_readvariableop@
<savev2_adam_batch_normalization_6_beta_m_read_readvariableop5
1savev2_adam_conv2d_5_kernel_m_read_readvariableop3
/savev2_adam_conv2d_5_bias_m_read_readvariableopA
=savev2_adam_batch_normalization_7_gamma_m_read_readvariableop@
<savev2_adam_batch_normalization_7_beta_m_read_readvariableop4
0savev2_adam_dense_3_kernel_m_read_readvariableop2
.savev2_adam_dense_3_bias_m_read_readvariableop4
0savev2_adam_dense_4_kernel_m_read_readvariableop2
.savev2_adam_dense_4_bias_m_read_readvariableop4
0savev2_adam_dense_5_kernel_m_read_readvariableop2
.savev2_adam_dense_5_bias_m_read_readvariableopA
=savev2_adam_batch_normalization_4_gamma_v_read_readvariableop@
<savev2_adam_batch_normalization_4_beta_v_read_readvariableop5
1savev2_adam_conv2d_3_kernel_v_read_readvariableop3
/savev2_adam_conv2d_3_bias_v_read_readvariableopA
=savev2_adam_batch_normalization_5_gamma_v_read_readvariableop@
<savev2_adam_batch_normalization_5_beta_v_read_readvariableop5
1savev2_adam_conv2d_4_kernel_v_read_readvariableop3
/savev2_adam_conv2d_4_bias_v_read_readvariableopA
=savev2_adam_batch_normalization_6_gamma_v_read_readvariableop@
<savev2_adam_batch_normalization_6_beta_v_read_readvariableop5
1savev2_adam_conv2d_5_kernel_v_read_readvariableop3
/savev2_adam_conv2d_5_bias_v_read_readvariableopA
=savev2_adam_batch_normalization_7_gamma_v_read_readvariableop@
<savev2_adam_batch_normalization_7_beta_v_read_readvariableop4
0savev2_adam_dense_3_kernel_v_read_readvariableop2
.savev2_adam_dense_3_bias_v_read_readvariableop4
0savev2_adam_dense_4_kernel_v_read_readvariableop2
.savev2_adam_dense_4_bias_v_read_readvariableop4
0savev2_adam_dense_5_kernel_v_read_readvariableop2
.savev2_adam_dense_5_bias_v_read_readvariableop
savev2_const

identity_1ИвMergeV2CheckpointsП
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
Const_1Л
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
ShardedFilename/shardж
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameО+
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:N*
dtype0*а*
valueЦ*BУ*NB5layer_with_weights-0/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-0/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-0/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesз
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:N*
dtype0*▒
valueзBдNB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesе!
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:06savev2_batch_normalization_4_gamma_read_readvariableop5savev2_batch_normalization_4_beta_read_readvariableop<savev2_batch_normalization_4_moving_mean_read_readvariableop@savev2_batch_normalization_4_moving_variance_read_readvariableop*savev2_conv2d_3_kernel_read_readvariableop(savev2_conv2d_3_bias_read_readvariableop6savev2_batch_normalization_5_gamma_read_readvariableop5savev2_batch_normalization_5_beta_read_readvariableop<savev2_batch_normalization_5_moving_mean_read_readvariableop@savev2_batch_normalization_5_moving_variance_read_readvariableop*savev2_conv2d_4_kernel_read_readvariableop(savev2_conv2d_4_bias_read_readvariableop6savev2_batch_normalization_6_gamma_read_readvariableop5savev2_batch_normalization_6_beta_read_readvariableop<savev2_batch_normalization_6_moving_mean_read_readvariableop@savev2_batch_normalization_6_moving_variance_read_readvariableop*savev2_conv2d_5_kernel_read_readvariableop(savev2_conv2d_5_bias_read_readvariableop6savev2_batch_normalization_7_gamma_read_readvariableop5savev2_batch_normalization_7_beta_read_readvariableop<savev2_batch_normalization_7_moving_mean_read_readvariableop@savev2_batch_normalization_7_moving_variance_read_readvariableop)savev2_dense_3_kernel_read_readvariableop'savev2_dense_3_bias_read_readvariableop)savev2_dense_4_kernel_read_readvariableop'savev2_dense_4_bias_read_readvariableop)savev2_dense_5_kernel_read_readvariableop'savev2_dense_5_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop=savev2_adam_batch_normalization_4_gamma_m_read_readvariableop<savev2_adam_batch_normalization_4_beta_m_read_readvariableop1savev2_adam_conv2d_3_kernel_m_read_readvariableop/savev2_adam_conv2d_3_bias_m_read_readvariableop=savev2_adam_batch_normalization_5_gamma_m_read_readvariableop<savev2_adam_batch_normalization_5_beta_m_read_readvariableop1savev2_adam_conv2d_4_kernel_m_read_readvariableop/savev2_adam_conv2d_4_bias_m_read_readvariableop=savev2_adam_batch_normalization_6_gamma_m_read_readvariableop<savev2_adam_batch_normalization_6_beta_m_read_readvariableop1savev2_adam_conv2d_5_kernel_m_read_readvariableop/savev2_adam_conv2d_5_bias_m_read_readvariableop=savev2_adam_batch_normalization_7_gamma_m_read_readvariableop<savev2_adam_batch_normalization_7_beta_m_read_readvariableop0savev2_adam_dense_3_kernel_m_read_readvariableop.savev2_adam_dense_3_bias_m_read_readvariableop0savev2_adam_dense_4_kernel_m_read_readvariableop.savev2_adam_dense_4_bias_m_read_readvariableop0savev2_adam_dense_5_kernel_m_read_readvariableop.savev2_adam_dense_5_bias_m_read_readvariableop=savev2_adam_batch_normalization_4_gamma_v_read_readvariableop<savev2_adam_batch_normalization_4_beta_v_read_readvariableop1savev2_adam_conv2d_3_kernel_v_read_readvariableop/savev2_adam_conv2d_3_bias_v_read_readvariableop=savev2_adam_batch_normalization_5_gamma_v_read_readvariableop<savev2_adam_batch_normalization_5_beta_v_read_readvariableop1savev2_adam_conv2d_4_kernel_v_read_readvariableop/savev2_adam_conv2d_4_bias_v_read_readvariableop=savev2_adam_batch_normalization_6_gamma_v_read_readvariableop<savev2_adam_batch_normalization_6_beta_v_read_readvariableop1savev2_adam_conv2d_5_kernel_v_read_readvariableop/savev2_adam_conv2d_5_bias_v_read_readvariableop=savev2_adam_batch_normalization_7_gamma_v_read_readvariableop<savev2_adam_batch_normalization_7_beta_v_read_readvariableop0savev2_adam_dense_3_kernel_v_read_readvariableop.savev2_adam_dense_3_bias_v_read_readvariableop0savev2_adam_dense_4_kernel_v_read_readvariableop.savev2_adam_dense_4_bias_v_read_readvariableop0savev2_adam_dense_5_kernel_v_read_readvariableop.savev2_adam_dense_5_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *\
dtypesR
P2N	2
SaveV2║
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesб
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

identity_1Identity_1:output:0*╓
_input_shapes─
┴: ::::::::::::::::: : : : : : :	рv@:@:@:::: : : : : : : : : ::::::::::: : : : :	рv@:@:@:::::::::::::: : : : :	рv@:@:@:::: 2(
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
:	рv@: 

_output_shapes
:@:$ 

_output_shapes

:@: 

_output_shapes
::$ 

_output_shapes

:: 
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
:	рv@: 5

_output_shapes
:@:$6 

_output_shapes

:@: 7

_output_shapes
::$8 

_output_shapes

:: 9
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
:	рv@: I

_output_shapes
:@:$J 

_output_shapes

:@: K

_output_shapes
::$L 

_output_shapes

:: M

_output_shapes
::N

_output_shapes
: 
·
}
(__inference_conv2d_5_layer_call_fn_46494

inputs
unknown
	unknown_0
identityИвStatefulPartitionedCall√
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         39 *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_conv2d_5_layer_call_and_return_conditional_losses_445362
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:         39 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:         39::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         39
 
_user_specified_nameinputs
╬
P
.__inference_stft_magnitude_layer_call_fn_43762
stft_1_input
identity╫
PartitionedCallPartitionedCallstft_1_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ╖Б* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_stft_magnitude_layer_call_and_return_conditional_losses_437592
PartitionedCallv
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:         ╖Б2

Identity"
identityIdentity:output:0*,
_input_shapes
:         Ат	:[ W
-
_output_shapes
:         Ат	
&
_user_specified_namestft_1_input
Ы
и
5__inference_batch_normalization_4_layer_call_fn_46101

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCall▓
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Y
fTRR
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_438242
StatefulPartitionedCallи
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
№
є
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_44388

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1╦
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:         gл:::::*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3█
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:         gл2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:         gл::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:         gл
 
_user_specified_nameinputs
А
f
J__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_43872

inputs
identityн
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
2	
MaxPoolЗ
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
я	
█
B__inference_dense_3_layer_call_and_return_conditional_losses_44651

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	рv@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:         @2
ReluЧ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         @2

Identity"
identityIdentity:output:0*/
_input_shapes
:         рv::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         рv
 
_user_specified_nameinputs
└
є
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_46384

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1▄
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3ь
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
ч
`
D__inference_reshape_1_layer_call_and_return_conditional_losses_44231

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
valueB	 :Ат	2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/2а
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapeu
ReshapeReshapeinputsReshape/shape:output:0*
T0*-
_output_shapes
:         Ат	2	
Reshapej
IdentityIdentityReshape:output:0*
T0*-
_output_shapes
:         Ат	2

Identity"
identityIdentity:output:0*(
_input_shapes
:         Ат	:Q M
)
_output_shapes
:         Ат	
 
_user_specified_nameinputs
╠
Ч
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_44172

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
FusedBatchNormV3н
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue╗
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1Р
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                            ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
╥X
└

G__inference_sequential_1_layer_call_and_return_conditional_losses_44782
reshape_1_input
batch_normalization_4_44314
batch_normalization_4_44316
batch_normalization_4_44318
batch_normalization_4_44320
conv2d_3_44345
conv2d_3_44347
batch_normalization_5_44415
batch_normalization_5_44417
batch_normalization_5_44419
batch_normalization_5_44421
conv2d_4_44446
conv2d_4_44448
batch_normalization_6_44516
batch_normalization_6_44518
batch_normalization_6_44520
batch_normalization_6_44522
conv2d_5_44547
conv2d_5_44549
batch_normalization_7_44617
batch_normalization_7_44619
batch_normalization_7_44621
batch_normalization_7_44623
dense_3_44662
dense_3_44664
dense_4_44719
dense_4_44721
dense_5_44776
dense_5_44778
identityИв-batch_normalization_4/StatefulPartitionedCallв-batch_normalization_5/StatefulPartitionedCallв-batch_normalization_6/StatefulPartitionedCallв-batch_normalization_7/StatefulPartitionedCallв conv2d_3/StatefulPartitionedCallв conv2d_4/StatefulPartitionedCallв conv2d_5/StatefulPartitionedCallвdense_3/StatefulPartitionedCallвdense_4/StatefulPartitionedCallвdense_5/StatefulPartitionedCallв!dropout_2/StatefulPartitionedCallв!dropout_3/StatefulPartitionedCallх
reshape_1/PartitionedCallPartitionedCallreshape_1_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:         Ат	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_reshape_1_layer_call_and_return_conditional_losses_442312
reshape_1/PartitionedCallЛ
stft_magnitude/PartitionedCallPartitionedCall"reshape_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ╖Б* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_stft_magnitude_layer_call_and_return_conditional_losses_437482 
stft_magnitude/PartitionedCall╣
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall'stft_magnitude/PartitionedCall:output:0batch_normalization_4_44314batch_normalization_4_44316batch_normalization_4_44318batch_normalization_4_44320*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ╖Б*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Y
fTRR
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_442692/
-batch_normalization_4/StatefulPartitionedCall╦
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0conv2d_3_44345conv2d_3_44347*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ╖Б*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_conv2d_3_layer_call_and_return_conditional_losses_443342"
 conv2d_3/StatefulPartitionedCallФ
max_pooling2d_3/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         gл* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_438722!
max_pooling2d_3/PartitionedCall╣
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_3/PartitionedCall:output:0batch_normalization_5_44415batch_normalization_5_44417batch_normalization_5_44419batch_normalization_5_44421*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         gл*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Y
fTRR
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_443702/
-batch_normalization_5/StatefulPartitionedCall╩
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0conv2d_4_44446conv2d_4_44448*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         gл*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_conv2d_4_layer_call_and_return_conditional_losses_444352"
 conv2d_4/StatefulPartitionedCallУ
max_pooling2d_4/PartitionedCallPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         39* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_439882!
max_pooling2d_4/PartitionedCall╕
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_4/PartitionedCall:output:0batch_normalization_6_44516batch_normalization_6_44518batch_normalization_6_44520batch_normalization_6_44522*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         39*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Y
fTRR
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_444712/
-batch_normalization_6/StatefulPartitionedCall╔
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:0conv2d_5_44547conv2d_5_44549*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         39 *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_conv2d_5_layer_call_and_return_conditional_losses_445362"
 conv2d_5/StatefulPartitionedCallУ
max_pooling2d_5/PartitionedCallPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_441042!
max_pooling2d_5/PartitionedCall╕
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_5/PartitionedCall:output:0batch_normalization_7_44617batch_normalization_7_44619batch_normalization_7_44621batch_normalization_7_44623*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Y
fTRR
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_445722/
-batch_normalization_7/StatefulPartitionedCallЗ
flatten_1/PartitionedCallPartitionedCall6batch_normalization_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         рv* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_446322
flatten_1/PartitionedCallи
dense_3/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_3_44662dense_3_44664*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_446512!
dense_3/StatefulPartitionedCallР
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_446792#
!dropout_2/StatefulPartitionedCall░
dense_4/StatefulPartitionedCallStatefulPartitionedCall*dropout_2/StatefulPartitionedCall:output:0dense_4_44719dense_4_44721*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_447082!
dense_4/StatefulPartitionedCall┤
!dropout_3/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0"^dropout_2/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dropout_3_layer_call_and_return_conditional_losses_447362#
!dropout_3/StatefulPartitionedCall░
dense_5/StatefulPartitionedCallStatefulPartitionedCall*dropout_3/StatefulPartitionedCall:output:0dense_5_44776dense_5_44778*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dense_5_layer_call_and_return_conditional_losses_447652!
dense_5/StatefulPartitionedCall╙
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall.^batch_normalization_6/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall"^dropout_3/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*Ъ
_input_shapesИ
Е:         Ат	::::::::::::::::::::::::::::2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2^
-batch_normalization_6/StatefulPartitionedCall-batch_normalization_6/StatefulPartitionedCall2^
-batch_normalization_7/StatefulPartitionedCall-batch_normalization_7/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall2F
!dropout_3/StatefulPartitionedCall!dropout_3/StatefulPartitionedCall:Z V
)
_output_shapes
:         Ат	
)
_user_specified_namereshape_1_input
╟
b
D__inference_dropout_2_layer_call_and_return_conditional_losses_44684

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:         @2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:         @2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:         @:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
ь
e
I__inference_stft_magnitude_layer_call_and_return_conditional_losses_43759

inputs
identity╫
stft_1/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ╖Б* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_stft_1_layer_call_and_return_conditional_losses_437112
stft_1/PartitionedCall 
magnitude_1/PartitionedCallPartitionedCallstft_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ╖Б* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_magnitude_1_layer_call_and_return_conditional_losses_437242
magnitude_1/PartitionedCallВ
IdentityIdentity$magnitude_1/PartitionedCall:output:0*
T0*1
_output_shapes
:         ╖Б2

Identity"
identityIdentity:output:0*,
_input_shapes
:         Ат	:U Q
-
_output_shapes
:         Ат	
 
_user_specified_nameinputs
И
Ч
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_46282

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1┘
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:         gл:::::*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
FusedBatchNormV3н
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue╗
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1 
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:         gл2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:         gл::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:         gл
 
_user_specified_nameinputs
╠
Ш
,__inference_sequential_1_layer_call_fn_45802

inputs
unknown
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

unknown_26
identityИвStatefulPartitionedCall┌
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
unknown_26*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *>
_read_only_resource_inputs 
	
*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_450802
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*Ъ
_input_shapesИ
Е:         Ат	::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
)
_output_shapes
:         Ат	
 
_user_specified_nameinputs
Ю
b
)__inference_dropout_2_layer_call_fn_46675

inputs
identityИвStatefulPartitionedCall┌
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_446792
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         @2

Identity"
identityIdentity:output:0*&
_input_shapes
:         @22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
╒
и
5__inference_batch_normalization_7_layer_call_fn_46622

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCallв
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Y
fTRR
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_445902
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:          2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:          ::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:          
 
_user_specified_nameinputs
дя
▐
G__inference_sequential_1_layer_call_and_return_conditional_losses_45680

inputs1
-batch_normalization_4_readvariableop_resource3
/batch_normalization_4_readvariableop_1_resourceB
>batch_normalization_4_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource+
'conv2d_3_conv2d_readvariableop_resource,
(conv2d_3_biasadd_readvariableop_resource1
-batch_normalization_5_readvariableop_resource3
/batch_normalization_5_readvariableop_1_resourceB
>batch_normalization_5_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource+
'conv2d_4_conv2d_readvariableop_resource,
(conv2d_4_biasadd_readvariableop_resource1
-batch_normalization_6_readvariableop_resource3
/batch_normalization_6_readvariableop_1_resourceB
>batch_normalization_6_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource+
'conv2d_5_conv2d_readvariableop_resource,
(conv2d_5_biasadd_readvariableop_resource1
-batch_normalization_7_readvariableop_resource3
/batch_normalization_7_readvariableop_1_resourceB
>batch_normalization_7_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
&dense_3_matmul_readvariableop_resource+
'dense_3_biasadd_readvariableop_resource*
&dense_4_matmul_readvariableop_resource+
'dense_4_biasadd_readvariableop_resource*
&dense_5_matmul_readvariableop_resource+
'dense_5_biasadd_readvariableop_resource
identityИв5batch_normalization_4/FusedBatchNormV3/ReadVariableOpв7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1в$batch_normalization_4/ReadVariableOpв&batch_normalization_4/ReadVariableOp_1в5batch_normalization_5/FusedBatchNormV3/ReadVariableOpв7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1в$batch_normalization_5/ReadVariableOpв&batch_normalization_5/ReadVariableOp_1в5batch_normalization_6/FusedBatchNormV3/ReadVariableOpв7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1в$batch_normalization_6/ReadVariableOpв&batch_normalization_6/ReadVariableOp_1в5batch_normalization_7/FusedBatchNormV3/ReadVariableOpв7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1в$batch_normalization_7/ReadVariableOpв&batch_normalization_7/ReadVariableOp_1вconv2d_3/BiasAdd/ReadVariableOpвconv2d_3/Conv2D/ReadVariableOpвconv2d_4/BiasAdd/ReadVariableOpвconv2d_4/Conv2D/ReadVariableOpвconv2d_5/BiasAdd/ReadVariableOpвconv2d_5/Conv2D/ReadVariableOpвdense_3/BiasAdd/ReadVariableOpвdense_3/MatMul/ReadVariableOpвdense_4/BiasAdd/ReadVariableOpвdense_4/MatMul/ReadVariableOpвdense_5/BiasAdd/ReadVariableOpвdense_5/MatMul/ReadVariableOpX
reshape_1/ShapeShapeinputs*
T0*
_output_shapes
:2
reshape_1/ShapeИ
reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_1/strided_slice/stackМ
reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_1/strided_slice/stack_1М
reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_1/strided_slice/stack_2Ю
reshape_1/strided_sliceStridedSlicereshape_1/Shape:output:0&reshape_1/strided_slice/stack:output:0(reshape_1/strided_slice/stack_1:output:0(reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_1/strided_slicez
reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB	 :Ат	2
reshape_1/Reshape/shape/1x
reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_1/Reshape/shape/2╥
reshape_1/Reshape/shapePack reshape_1/strided_slice:output:0"reshape_1/Reshape/shape/1:output:0"reshape_1/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape_1/Reshape/shapeУ
reshape_1/ReshapeReshapeinputs reshape_1/Reshape/shape:output:0*
T0*-
_output_shapes
:         Ат	2
reshape_1/Reshapeб
$stft_magnitude/stft_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2&
$stft_magnitude/stft_1/transpose/perm╥
stft_magnitude/stft_1/transpose	Transposereshape_1/Reshape:output:0-stft_magnitude/stft_1/transpose/perm:output:0*
T0*-
_output_shapes
:         Ат	2!
stft_magnitude/stft_1/transpose╖
8stft_magnitude/stft_1/stft_1_tf.signal.stft/frame_lengthConst*
_output_shapes
: *
dtype0*
value
B :А2:
8stft_magnitude/stft_1/stft_1_tf.signal.stft/frame_length│
6stft_magnitude/stft_1/stft_1_tf.signal.stft/frame_stepConst*
_output_shapes
: *
dtype0*
value
B :А28
6stft_magnitude/stft_1/stft_1_tf.signal.stft/frame_step│
6stft_magnitude/stft_1/stft_1_tf.signal.stft/fft_lengthConst*
_output_shapes
: *
dtype0*
value
B :А28
6stft_magnitude/stft_1/stft_1_tf.signal.stft/fft_length╗
6stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/axisConst*
_output_shapes
: *
dtype0*
valueB :
         28
6stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/axis┼
7stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/ShapeShape#stft_magnitude/stft_1/transpose:y:0*
T0*
_output_shapes
:29
7stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/Shape▓
6stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/RankConst*
_output_shapes
: *
dtype0*
value	B :28
6stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/Rank└
=stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2?
=stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/range/start└
=stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2?
=stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/range/deltaш
7stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/rangeRangeFstft_magnitude/stft_1/stft_1_tf.signal.stft/frame/range/start:output:0?stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/Rank:output:0Fstft_magnitude/stft_1/stft_1_tf.signal.stft/frame/range/delta:output:0*
_output_shapes
:29
7stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/rangeс
Estft_magnitude/stft_1/stft_1_tf.signal.stft/frame/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2G
Estft_magnitude/stft_1/stft_1_tf.signal.stft/frame/strided_slice/stack▄
Gstft_magnitude/stft_1/stft_1_tf.signal.stft/frame/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2I
Gstft_magnitude/stft_1/stft_1_tf.signal.stft/frame/strided_slice/stack_1▄
Gstft_magnitude/stft_1/stft_1_tf.signal.stft/frame/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2I
Gstft_magnitude/stft_1/stft_1_tf.signal.stft/frame/strided_slice/stack_2О
?stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/strided_sliceStridedSlice@stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/range:output:0Nstft_magnitude/stft_1/stft_1_tf.signal.stft/frame/strided_slice/stack:output:0Pstft_magnitude/stft_1/stft_1_tf.signal.stft/frame/strided_slice/stack_1:output:0Pstft_magnitude/stft_1/stft_1_tf.signal.stft/frame/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2A
?stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/strided_slice┤
7stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/sub/yConst*
_output_shapes
: *
dtype0*
value	B :29
7stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/sub/yЩ
5stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/subSub?stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/Rank:output:0@stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/sub/y:output:0*
T0*
_output_shapes
: 27
5stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/subЯ
7stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/sub_1Sub9stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/sub:z:0Hstft_magnitude/stft_1/stft_1_tf.signal.stft/frame/strided_slice:output:0*
T0*
_output_shapes
: 29
7stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/sub_1║
:stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2<
:stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/packed/1Ў
8stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/packedPackHstft_magnitude/stft_1/stft_1_tf.signal.stft/frame/strided_slice:output:0Cstft_magnitude/stft_1/stft_1_tf.signal.stft/frame/packed/1:output:0;stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/sub_1:z:0*
N*
T0*
_output_shapes
:2:
8stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/packed╚
Astft_magnitude/stft_1/stft_1_tf.signal.stft/frame/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2C
Astft_magnitude/stft_1/stft_1_tf.signal.stft/frame/split/split_dimЩ
7stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/splitSplitV@stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/Shape:output:0Astft_magnitude/stft_1/stft_1_tf.signal.stft/frame/packed:output:0Jstft_magnitude/stft_1/stft_1_tf.signal.stft/frame/split/split_dim:output:0*
T0*

Tlen0*$
_output_shapes
::: *
	num_split29
7stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/split┼
?stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/Reshape/shapeConst*
_output_shapes
: *
dtype0*
valueB 2A
?stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/Reshape/shape╔
Astft_magnitude/stft_1/stft_1_tf.signal.stft/frame/Reshape/shape_1Const*
_output_shapes
: *
dtype0*
valueB 2C
Astft_magnitude/stft_1/stft_1_tf.signal.stft/frame/Reshape/shape_1░
9stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/ReshapeReshape@stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/split:output:1Jstft_magnitude/stft_1/stft_1_tf.signal.stft/frame/Reshape/shape_1:output:0*
T0*
_output_shapes
: 2;
9stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/Reshape▓
6stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/SizeConst*
_output_shapes
: *
dtype0*
value	B :28
6stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/Size╢
8stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/Size_1Const*
_output_shapes
: *
dtype0*
value	B : 2:
8stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/Size_1б
7stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/sub_2SubBstft_magnitude/stft_1/stft_1_tf.signal.stft/frame/Reshape:output:0Astft_magnitude/stft_1/stft_1_tf.signal.stft/frame_length:output:0*
T0*
_output_shapes
: 29
7stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/sub_2г
:stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/floordivFloorDiv;stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/sub_2:z:0?stft_magnitude/stft_1/stft_1_tf.signal.stft/frame_step:output:0*
T0*
_output_shapes
: 2<
:stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/floordiv┤
7stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/add/xConst*
_output_shapes
: *
dtype0*
value	B :29
7stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/add/xЪ
5stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/addAddV2@stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/add/x:output:0>stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/floordiv:z:0*
T0*
_output_shapes
: 27
5stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/add╝
;stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/Maximum/xConst*
_output_shapes
: *
dtype0*
value	B : 2=
;stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/Maximum/xг
9stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/MaximumMaximumDstft_magnitude/stft_1/stft_1_tf.signal.stft/frame/Maximum/x:output:09stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/add:z:0*
T0*
_output_shapes
: 2;
9stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/Maximum╜
;stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/gcd/ConstConst*
_output_shapes
: *
dtype0*
value
B :А2=
;stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/gcd/Const├
>stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/floordiv_1/yConst*
_output_shapes
: *
dtype0*
value
B :А2@
>stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/floordiv_1/y╡
<stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/floordiv_1FloorDivAstft_magnitude/stft_1/stft_1_tf.signal.stft/frame_length:output:0Gstft_magnitude/stft_1/stft_1_tf.signal.stft/frame/floordiv_1/y:output:0*
T0*
_output_shapes
: 2>
<stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/floordiv_1├
>stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/floordiv_2/yConst*
_output_shapes
: *
dtype0*
value
B :А2@
>stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/floordiv_2/y│
<stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/floordiv_2FloorDiv?stft_magnitude/stft_1/stft_1_tf.signal.stft/frame_step:output:0Gstft_magnitude/stft_1/stft_1_tf.signal.stft/frame/floordiv_2/y:output:0*
T0*
_output_shapes
: 2>
<stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/floordiv_2├
>stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/floordiv_3/yConst*
_output_shapes
: *
dtype0*
value
B :А2@
>stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/floordiv_3/y╢
<stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/floordiv_3FloorDivBstft_magnitude/stft_1/stft_1_tf.signal.stft/frame/Reshape:output:0Gstft_magnitude/stft_1/stft_1_tf.signal.stft/frame/floordiv_3/y:output:0*
T0*
_output_shapes
: 2>
<stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/floordiv_3╡
7stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/mul/yConst*
_output_shapes
: *
dtype0*
value
B :А29
7stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/mul/yЪ
5stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/mulMul@stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/floordiv_3:z:0@stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/mul/y:output:0*
T0*
_output_shapes
: 27
5stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/mulў
Astft_magnitude/stft_1/stft_1_tf.signal.stft/frame/concat/values_1Pack9stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/mul:z:0*
N*
T0*
_output_shapes
:2C
Astft_magnitude/stft_1/stft_1_tf.signal.stft/frame/concat/values_1└
=stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2?
=stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/concat/axis╞
8stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/concatConcatV2@stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/split:output:0Jstft_magnitude/stft_1/stft_1_tf.signal.stft/frame/concat/values_1:output:0@stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/split:output:2Fstft_magnitude/stft_1/stft_1_tf.signal.stft/frame/concat/axis:output:0*
N*
T0*
_output_shapes
:2:
8stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/concat╤
Estft_magnitude/stft_1/stft_1_tf.signal.stft/frame/concat_1/values_1/1Const*
_output_shapes
: *
dtype0*
value
B :А2G
Estft_magnitude/stft_1/stft_1_tf.signal.stft/frame/concat_1/values_1/1╥
Cstft_magnitude/stft_1/stft_1_tf.signal.stft/frame/concat_1/values_1Pack@stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/floordiv_3:z:0Nstft_magnitude/stft_1/stft_1_tf.signal.stft/frame/concat_1/values_1/1:output:0*
N*
T0*
_output_shapes
:2E
Cstft_magnitude/stft_1/stft_1_tf.signal.stft/frame/concat_1/values_1─
?stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2A
?stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/concat_1/axis╬
:stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/concat_1ConcatV2@stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/split:output:0Lstft_magnitude/stft_1/stft_1_tf.signal.stft/frame/concat_1/values_1:output:0@stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/split:output:2Hstft_magnitude/stft_1/stft_1_tf.signal.stft/frame/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2<
:stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/concat_1╞
<stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/zeros_likeConst*
_output_shapes
:*
dtype0*
valueB: 2>
<stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/zeros_like╨
Astft_magnitude/stft_1/stft_1_tf.signal.stft/frame/ones_like/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2C
Astft_magnitude/stft_1/stft_1_tf.signal.stft/frame/ones_like/Shape╚
Astft_magnitude/stft_1/stft_1_tf.signal.stft/frame/ones_like/ConstConst*
_output_shapes
: *
dtype0*
value	B :2C
Astft_magnitude/stft_1/stft_1_tf.signal.stft/frame/ones_like/Const┐
;stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/ones_likeFillJstft_magnitude/stft_1/stft_1_tf.signal.stft/frame/ones_like/Shape:output:0Jstft_magnitude/stft_1/stft_1_tf.signal.stft/frame/ones_like/Const:output:0*
T0*
_output_shapes
:2=
;stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/ones_like┌
>stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/StridedSliceStridedSlice#stft_magnitude/stft_1/transpose:y:0Estft_magnitude/stft_1/stft_1_tf.signal.stft/frame/zeros_like:output:0Astft_magnitude/stft_1/stft_1_tf.signal.stft/frame/concat:output:0Dstft_magnitude/stft_1/stft_1_tf.signal.stft/frame/ones_like:output:0*
Index0*
T0*=
_output_shapes+
):'                           2@
>stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/StridedSliceр
;stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/Reshape_1ReshapeGstft_magnitude/stft_1/stft_1_tf.signal.stft/frame/StridedSlice:output:0Cstft_magnitude/stft_1/stft_1_tf.signal.stft/frame/concat_1:output:0*
T0*B
_output_shapes0
.:,                           А2=
;stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/Reshape_1─
?stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/range_1/startConst*
_output_shapes
: *
dtype0*
value	B : 2A
?stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/range_1/start─
?stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :2A
?stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/range_1/deltaў
9stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/range_1RangeHstft_magnitude/stft_1/stft_1_tf.signal.stft/frame/range_1/start:output:0=stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/Maximum:z:0Hstft_magnitude/stft_1/stft_1_tf.signal.stft/frame/range_1/delta:output:0*#
_output_shapes
:         2;
9stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/range_1н
7stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/mul_1MulBstft_magnitude/stft_1/stft_1_tf.signal.stft/frame/range_1:output:0@stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/floordiv_2:z:0*
T0*#
_output_shapes
:         29
7stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/mul_1╠
Cstft_magnitude/stft_1/stft_1_tf.signal.stft/frame/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2E
Cstft_magnitude/stft_1/stft_1_tf.signal.stft/frame/Reshape_2/shape/1╔
Astft_magnitude/stft_1/stft_1_tf.signal.stft/frame/Reshape_2/shapePack=stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/Maximum:z:0Lstft_magnitude/stft_1/stft_1_tf.signal.stft/frame/Reshape_2/shape/1:output:0*
N*
T0*
_output_shapes
:2C
Astft_magnitude/stft_1/stft_1_tf.signal.stft/frame/Reshape_2/shape└
;stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/Reshape_2Reshape;stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/mul_1:z:0Jstft_magnitude/stft_1/stft_1_tf.signal.stft/frame/Reshape_2/shape:output:0*
T0*'
_output_shapes
:         2=
;stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/Reshape_2─
?stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/range_2/startConst*
_output_shapes
: *
dtype0*
value	B : 2A
?stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/range_2/start─
?stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/range_2/deltaConst*
_output_shapes
: *
dtype0*
value	B :2A
?stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/range_2/deltaё
9stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/range_2RangeHstft_magnitude/stft_1/stft_1_tf.signal.stft/frame/range_2/start:output:0@stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/floordiv_1:z:0Hstft_magnitude/stft_1/stft_1_tf.signal.stft/frame/range_2/delta:output:0*
_output_shapes
:2;
9stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/range_2╠
Cstft_magnitude/stft_1/stft_1_tf.signal.stft/frame/Reshape_3/shape/0Const*
_output_shapes
: *
dtype0*
value	B :2E
Cstft_magnitude/stft_1/stft_1_tf.signal.stft/frame/Reshape_3/shape/0╠
Astft_magnitude/stft_1/stft_1_tf.signal.stft/frame/Reshape_3/shapePackLstft_magnitude/stft_1/stft_1_tf.signal.stft/frame/Reshape_3/shape/0:output:0@stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/floordiv_1:z:0*
N*
T0*
_output_shapes
:2C
Astft_magnitude/stft_1/stft_1_tf.signal.stft/frame/Reshape_3/shape╛
;stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/Reshape_3ReshapeBstft_magnitude/stft_1/stft_1_tf.signal.stft/frame/range_2:output:0Jstft_magnitude/stft_1/stft_1_tf.signal.stft/frame/Reshape_3/shape:output:0*
T0*
_output_shapes

:2=
;stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/Reshape_3╣
7stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/add_1AddV2Dstft_magnitude/stft_1/stft_1_tf.signal.stft/frame/Reshape_2:output:0Dstft_magnitude/stft_1/stft_1_tf.signal.stft/frame/Reshape_3:output:0*
T0*'
_output_shapes
:         29
7stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/add_1┼
:stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/GatherV2GatherV2Dstft_magnitude/stft_1/stft_1_tf.signal.stft/frame/Reshape_1:output:0;stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/add_1:z:0Hstft_magnitude/stft_1/stft_1_tf.signal.stft/frame/strided_slice:output:0*
Taxis0*
Tindices0*
Tparams0*F
_output_shapes4
2:0                           А2<
:stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/GatherV2┬
Cstft_magnitude/stft_1/stft_1_tf.signal.stft/frame/concat_2/values_1Pack=stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/Maximum:z:0Astft_magnitude/stft_1/stft_1_tf.signal.stft/frame_length:output:0*
N*
T0*
_output_shapes
:2E
Cstft_magnitude/stft_1/stft_1_tf.signal.stft/frame/concat_2/values_1─
?stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2A
?stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/concat_2/axis╬
:stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/concat_2ConcatV2@stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/split:output:0Lstft_magnitude/stft_1/stft_1_tf.signal.stft/frame/concat_2/values_1:output:0@stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/split:output:2Hstft_magnitude/stft_1/stft_1_tf.signal.stft/frame/concat_2/axis:output:0*
N*
T0*
_output_shapes
:2<
:stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/concat_2╦
;stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/Reshape_4ReshapeCstft_magnitude/stft_1/stft_1_tf.signal.stft/frame/GatherV2:output:0Cstft_magnitude/stft_1/stft_1_tf.signal.stft/frame/concat_2:output:0*
T0*1
_output_shapes
:         ╖А2=
;stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/Reshape_4╞
@stft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/periodicConst*
_output_shapes
: *
dtype0
*
value	B
 Z2B
@stft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/periodic 
<stft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/CastCastIstft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/periodic:output:0*

DstT0*

SrcT0
*
_output_shapes
: 2>
<stft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/Cast╩
Bstft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/FloorMod/yConst*
_output_shapes
: *
dtype0*
value	B :2D
Bstft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/FloorMod/y┴
@stft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/FloorModFloorModAstft_magnitude/stft_1/stft_1_tf.signal.stft/frame_length:output:0Kstft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/FloorMod/y:output:0*
T0*
_output_shapes
: 2B
@stft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/FloorMod└
=stft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/sub/xConst*
_output_shapes
: *
dtype0*
value	B :2?
=stft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/sub/x░
;stft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/subSubFstft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/sub/x:output:0Dstft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/FloorMod:z:0*
T0*
_output_shapes
: 2=
;stft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/subе
;stft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/mulMul@stft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/Cast:y:0?stft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/sub:z:0*
T0*
_output_shapes
: 2=
;stft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/mulи
;stft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/addAddV2Astft_magnitude/stft_1/stft_1_tf.signal.stft/frame_length:output:0?stft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/mul:z:0*
T0*
_output_shapes
: 2=
;stft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/add─
?stft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/sub_1/yConst*
_output_shapes
: *
dtype0*
value	B :2A
?stft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/sub_1/y▒
=stft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/sub_1Sub?stft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/add:z:0Hstft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/sub_1/y:output:0*
T0*
_output_shapes
: 2?
=stft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/sub_1√
>stft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/Cast_1CastAstft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/sub_1:z:0*

DstT0*

SrcT0*
_output_shapes
: 2@
>stft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/Cast_1╠
Cstft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2E
Cstft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/range/start╠
Cstft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2E
Cstft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/range/deltaГ
=stft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/rangeRangeLstft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/range/start:output:0Astft_magnitude/stft_1/stft_1_tf.signal.stft/frame_length:output:0Lstft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/range/delta:output:0*
_output_shapes	
:А2?
=stft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/rangeЕ
>stft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/Cast_2CastFstft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/range:output:0*

DstT0*

SrcT0*
_output_shapes	
:А2@
>stft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/Cast_2├
=stft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *█╔@2?
=stft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/Const╖
=stft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/mul_1MulFstft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/Const:output:0Bstft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/Cast_2:y:0*
T0*
_output_shapes	
:А2?
=stft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/mul_1║
?stft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/truedivRealDivAstft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/mul_1:z:0Bstft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/Cast_1:y:0*
T0*
_output_shapes	
:А2A
?stft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/truedivь
;stft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/CosCosCstft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/truediv:z:0*
T0*
_output_shapes	
:А2=
;stft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/Cos╟
?stft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2A
?stft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/mul_2/x╢
=stft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/mul_2MulHstft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/mul_2/x:output:0?stft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/Cos:y:0*
T0*
_output_shapes	
:А2?
=stft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/mul_2╟
?stft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2A
?stft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/sub_2/x╕
=stft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/sub_2SubHstft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/sub_2/x:output:0Astft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/mul_2:z:0*
T0*
_output_shapes	
:А2?
=stft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/sub_2о
/stft_magnitude/stft_1/stft_1_tf.signal.stft/mulMulDstft_magnitude/stft_1/stft_1_tf.signal.stft/frame/Reshape_4:output:0Astft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/sub_2:z:0*
T0*1
_output_shapes
:         ╖А21
/stft_magnitude/stft_1/stft_1_tf.signal.stft/mulщ
7stft_magnitude/stft_1/stft_1_tf.signal.stft/rfft/packedPack?stft_magnitude/stft_1/stft_1_tf.signal.stft/fft_length:output:0*
N*
T0*
_output_shapes
:29
7stft_magnitude/stft_1/stft_1_tf.signal.stft/rfft/packed┼
;stft_magnitude/stft_1/stft_1_tf.signal.stft/rfft/fft_lengthConst*
_output_shapes
:*
dtype0*
valueB:А2=
;stft_magnitude/stft_1/stft_1_tf.signal.stft/rfft/fft_lengthЪ
0stft_magnitude/stft_1/stft_1_tf.signal.stft/rfftRFFT3stft_magnitude/stft_1/stft_1_tf.signal.stft/mul:z:0Dstft_magnitude/stft_1/stft_1_tf.signal.stft/rfft/fft_length:output:0*1
_output_shapes
:         ╖Б22
0stft_magnitude/stft_1/stft_1_tf.signal.stft/rfftй
&stft_magnitude/stft_1/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             2(
&stft_magnitude/stft_1/transpose_1/perm√
!stft_magnitude/stft_1/transpose_1	Transpose9stft_magnitude/stft_1/stft_1_tf.signal.stft/rfft:output:0/stft_magnitude/stft_1/transpose_1/perm:output:0*
T0*1
_output_shapes
:         ╖Б2#
!stft_magnitude/stft_1/transpose_1и
stft_magnitude/magnitude_1/Abs
ComplexAbs%stft_magnitude/stft_1/transpose_1:y:0*1
_output_shapes
:         ╖Б2 
stft_magnitude/magnitude_1/Abs╢
$batch_normalization_4/ReadVariableOpReadVariableOp-batch_normalization_4_readvariableop_resource*
_output_shapes
:*
dtype02&
$batch_normalization_4/ReadVariableOp╝
&batch_normalization_4/ReadVariableOp_1ReadVariableOp/batch_normalization_4_readvariableop_1_resource*
_output_shapes
:*
dtype02(
&batch_normalization_4/ReadVariableOp_1щ
5batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_4/FusedBatchNormV3/ReadVariableOpя
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype029
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ь
&batch_normalization_4/FusedBatchNormV3FusedBatchNormV3"stft_magnitude/magnitude_1/Abs:y:0,batch_normalization_4/ReadVariableOp:value:0.batch_normalization_4/ReadVariableOp_1:value:0=batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:         ╖Б:::::*
epsilon%oГ:*
is_training( 2(
&batch_normalization_4/FusedBatchNormV3░
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02 
conv2d_3/Conv2D/ReadVariableOpф
conv2d_3/Conv2DConv2D*batch_normalization_4/FusedBatchNormV3:y:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ╖Б*
paddingSAME*
strides
2
conv2d_3/Conv2Dз
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_3/BiasAdd/ReadVariableOpо
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ╖Б2
conv2d_3/BiasAdd}
conv2d_3/ReluReluconv2d_3/BiasAdd:output:0*
T0*1
_output_shapes
:         ╖Б2
conv2d_3/Relu╚
max_pooling2d_3/MaxPoolMaxPoolconv2d_3/Relu:activations:0*0
_output_shapes
:         gл*
ksize
*
paddingVALID*
strides
2
max_pooling2d_3/MaxPool╢
$batch_normalization_5/ReadVariableOpReadVariableOp-batch_normalization_5_readvariableop_resource*
_output_shapes
:*
dtype02&
$batch_normalization_5/ReadVariableOp╝
&batch_normalization_5/ReadVariableOp_1ReadVariableOp/batch_normalization_5_readvariableop_1_resource*
_output_shapes
:*
dtype02(
&batch_normalization_5/ReadVariableOp_1щ
5batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_5/FusedBatchNormV3/ReadVariableOpя
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype029
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1щ
&batch_normalization_5/FusedBatchNormV3FusedBatchNormV3 max_pooling2d_3/MaxPool:output:0,batch_normalization_5/ReadVariableOp:value:0.batch_normalization_5/ReadVariableOp_1:value:0=batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:         gл:::::*
epsilon%oГ:*
is_training( 2(
&batch_normalization_5/FusedBatchNormV3░
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02 
conv2d_4/Conv2D/ReadVariableOpу
conv2d_4/Conv2DConv2D*batch_normalization_5/FusedBatchNormV3:y:0&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         gл*
paddingSAME*
strides
2
conv2d_4/Conv2Dз
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_4/BiasAdd/ReadVariableOpн
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         gл2
conv2d_4/BiasAdd|
conv2d_4/ReluReluconv2d_4/BiasAdd:output:0*
T0*0
_output_shapes
:         gл2
conv2d_4/Relu╟
max_pooling2d_4/MaxPoolMaxPoolconv2d_4/Relu:activations:0*/
_output_shapes
:         39*
ksize
*
paddingVALID*
strides
2
max_pooling2d_4/MaxPool╢
$batch_normalization_6/ReadVariableOpReadVariableOp-batch_normalization_6_readvariableop_resource*
_output_shapes
:*
dtype02&
$batch_normalization_6/ReadVariableOp╝
&batch_normalization_6/ReadVariableOp_1ReadVariableOp/batch_normalization_6_readvariableop_1_resource*
_output_shapes
:*
dtype02(
&batch_normalization_6/ReadVariableOp_1щ
5batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_6/FusedBatchNormV3/ReadVariableOpя
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype029
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ш
&batch_normalization_6/FusedBatchNormV3FusedBatchNormV3 max_pooling2d_4/MaxPool:output:0,batch_normalization_6/ReadVariableOp:value:0.batch_normalization_6/ReadVariableOp_1:value:0=batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         39:::::*
epsilon%oГ:*
is_training( 2(
&batch_normalization_6/FusedBatchNormV3░
conv2d_5/Conv2D/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02 
conv2d_5/Conv2D/ReadVariableOpт
conv2d_5/Conv2DConv2D*batch_normalization_6/FusedBatchNormV3:y:0&conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         39 *
paddingSAME*
strides
2
conv2d_5/Conv2Dз
conv2d_5/BiasAdd/ReadVariableOpReadVariableOp(conv2d_5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_5/BiasAdd/ReadVariableOpм
conv2d_5/BiasAddBiasAddconv2d_5/Conv2D:output:0'conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         39 2
conv2d_5/BiasAdd{
conv2d_5/ReluReluconv2d_5/BiasAdd:output:0*
T0*/
_output_shapes
:         39 2
conv2d_5/Relu╟
max_pooling2d_5/MaxPoolMaxPoolconv2d_5/Relu:activations:0*/
_output_shapes
:          *
ksize
*
paddingVALID*
strides
2
max_pooling2d_5/MaxPool╢
$batch_normalization_7/ReadVariableOpReadVariableOp-batch_normalization_7_readvariableop_resource*
_output_shapes
: *
dtype02&
$batch_normalization_7/ReadVariableOp╝
&batch_normalization_7/ReadVariableOp_1ReadVariableOp/batch_normalization_7_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&batch_normalization_7/ReadVariableOp_1щ
5batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype027
5batch_normalization_7/FusedBatchNormV3/ReadVariableOpя
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype029
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ш
&batch_normalization_7/FusedBatchNormV3FusedBatchNormV3 max_pooling2d_5/MaxPool:output:0,batch_normalization_7/ReadVariableOp:value:0.batch_normalization_7/ReadVariableOp_1:value:0=batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:          : : : : :*
epsilon%oГ:*
is_training( 2(
&batch_normalization_7/FusedBatchNormV3s
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"    `;  2
flatten_1/Constк
flatten_1/ReshapeReshape*batch_normalization_7/FusedBatchNormV3:y:0flatten_1/Const:output:0*
T0*(
_output_shapes
:         рv2
flatten_1/Reshapeж
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes
:	рv@*
dtype02
dense_3/MatMul/ReadVariableOpЯ
dense_3/MatMulMatMulflatten_1/Reshape:output:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2
dense_3/MatMulд
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
dense_3/BiasAdd/ReadVariableOpб
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2
dense_3/BiasAddp
dense_3/ReluReludense_3/BiasAdd:output:0*
T0*'
_output_shapes
:         @2
dense_3/ReluВ
dropout_2/IdentityIdentitydense_3/Relu:activations:0*
T0*'
_output_shapes
:         @2
dropout_2/Identityе
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02
dense_4/MatMul/ReadVariableOpа
dense_4/MatMulMatMuldropout_2/Identity:output:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_4/MatMulд
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_4/BiasAdd/ReadVariableOpб
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_4/BiasAddp
dense_4/ReluReludense_4/BiasAdd:output:0*
T0*'
_output_shapes
:         2
dense_4/ReluВ
dropout_3/IdentityIdentitydense_4/Relu:activations:0*
T0*'
_output_shapes
:         2
dropout_3/Identityе
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_5/MatMul/ReadVariableOpа
dense_5/MatMulMatMuldropout_3/Identity:output:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_5/MatMulд
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_5/BiasAdd/ReadVariableOpб
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_5/BiasAddy
dense_5/SigmoidSigmoiddense_5/BiasAdd:output:0*
T0*'
_output_shapes
:         2
dense_5/Sigmoid√	
IdentityIdentitydense_5/Sigmoid:y:06^batch_normalization_4/FusedBatchNormV3/ReadVariableOp8^batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_4/ReadVariableOp'^batch_normalization_4/ReadVariableOp_16^batch_normalization_5/FusedBatchNormV3/ReadVariableOp8^batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_5/ReadVariableOp'^batch_normalization_5/ReadVariableOp_16^batch_normalization_6/FusedBatchNormV3/ReadVariableOp8^batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_6/ReadVariableOp'^batch_normalization_6/ReadVariableOp_16^batch_normalization_7/FusedBatchNormV3/ReadVariableOp8^batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_7/ReadVariableOp'^batch_normalization_7/ReadVariableOp_1 ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp ^conv2d_4/BiasAdd/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp ^conv2d_5/BiasAdd/ReadVariableOp^conv2d_5/Conv2D/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*Ъ
_input_shapesИ
Е:         Ат	::::::::::::::::::::::::::::2n
5batch_normalization_4/FusedBatchNormV3/ReadVariableOp5batch_normalization_4/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_17batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_4/ReadVariableOp$batch_normalization_4/ReadVariableOp2P
&batch_normalization_4/ReadVariableOp_1&batch_normalization_4/ReadVariableOp_12n
5batch_normalization_5/FusedBatchNormV3/ReadVariableOp5batch_normalization_5/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_17batch_normalization_5/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_5/ReadVariableOp$batch_normalization_5/ReadVariableOp2P
&batch_normalization_5/ReadVariableOp_1&batch_normalization_5/ReadVariableOp_12n
5batch_normalization_6/FusedBatchNormV3/ReadVariableOp5batch_normalization_6/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_17batch_normalization_6/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_6/ReadVariableOp$batch_normalization_6/ReadVariableOp2P
&batch_normalization_6/ReadVariableOp_1&batch_normalization_6/ReadVariableOp_12n
5batch_normalization_7/FusedBatchNormV3/ReadVariableOp5batch_normalization_7/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_17batch_normalization_7/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_7/ReadVariableOp$batch_normalization_7/ReadVariableOp2P
&batch_normalization_7/ReadVariableOp_1&batch_normalization_7/ReadVariableOp_12B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2B
conv2d_4/BiasAdd/ReadVariableOpconv2d_4/BiasAdd/ReadVariableOp2@
conv2d_4/Conv2D/ReadVariableOpconv2d_4/Conv2D/ReadVariableOp2B
conv2d_5/BiasAdd/ReadVariableOpconv2d_5/BiasAdd/ReadVariableOp2@
conv2d_5/Conv2D/ReadVariableOpconv2d_5/Conv2D/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp:Q M
)
_output_shapes
:         Ат	
 
_user_specified_nameinputs
Д
Ч
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_44572

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1╪
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:          : : : : :*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
FusedBatchNormV3н
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue╗
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1■
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:          2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:          ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:          
 
_user_specified_nameinputs
▌
и
5__inference_batch_normalization_4_layer_call_fn_46178

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCallд
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ╖Б*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Y
fTRR
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_442872
StatefulPartitionedCallШ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:         ╖Б2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:         ╖Б::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:         ╖Б
 
_user_specified_nameinputs
Т
E
)__inference_dropout_3_layer_call_fn_46727

inputs
identity┬
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dropout_3_layer_call_and_return_conditional_losses_447412
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
╟
b
D__inference_dropout_2_layer_call_and_return_conditional_losses_46670

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:         @2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:         @2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:         @:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
└
є
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_46088

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1▄
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3ь
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
╖X
╖

G__inference_sequential_1_layer_call_and_return_conditional_losses_44941

inputs
batch_normalization_4_44868
batch_normalization_4_44870
batch_normalization_4_44872
batch_normalization_4_44874
conv2d_3_44877
conv2d_3_44879
batch_normalization_5_44883
batch_normalization_5_44885
batch_normalization_5_44887
batch_normalization_5_44889
conv2d_4_44892
conv2d_4_44894
batch_normalization_6_44898
batch_normalization_6_44900
batch_normalization_6_44902
batch_normalization_6_44904
conv2d_5_44907
conv2d_5_44909
batch_normalization_7_44913
batch_normalization_7_44915
batch_normalization_7_44917
batch_normalization_7_44919
dense_3_44923
dense_3_44925
dense_4_44929
dense_4_44931
dense_5_44935
dense_5_44937
identityИв-batch_normalization_4/StatefulPartitionedCallв-batch_normalization_5/StatefulPartitionedCallв-batch_normalization_6/StatefulPartitionedCallв-batch_normalization_7/StatefulPartitionedCallв conv2d_3/StatefulPartitionedCallв conv2d_4/StatefulPartitionedCallв conv2d_5/StatefulPartitionedCallвdense_3/StatefulPartitionedCallвdense_4/StatefulPartitionedCallвdense_5/StatefulPartitionedCallв!dropout_2/StatefulPartitionedCallв!dropout_3/StatefulPartitionedCall▄
reshape_1/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:         Ат	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_reshape_1_layer_call_and_return_conditional_losses_442312
reshape_1/PartitionedCallЛ
stft_magnitude/PartitionedCallPartitionedCall"reshape_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ╖Б* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_stft_magnitude_layer_call_and_return_conditional_losses_437482 
stft_magnitude/PartitionedCall╣
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall'stft_magnitude/PartitionedCall:output:0batch_normalization_4_44868batch_normalization_4_44870batch_normalization_4_44872batch_normalization_4_44874*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ╖Б*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Y
fTRR
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_442692/
-batch_normalization_4/StatefulPartitionedCall╦
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0conv2d_3_44877conv2d_3_44879*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ╖Б*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_conv2d_3_layer_call_and_return_conditional_losses_443342"
 conv2d_3/StatefulPartitionedCallФ
max_pooling2d_3/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         gл* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_438722!
max_pooling2d_3/PartitionedCall╣
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_3/PartitionedCall:output:0batch_normalization_5_44883batch_normalization_5_44885batch_normalization_5_44887batch_normalization_5_44889*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         gл*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Y
fTRR
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_443702/
-batch_normalization_5/StatefulPartitionedCall╩
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0conv2d_4_44892conv2d_4_44894*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         gл*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_conv2d_4_layer_call_and_return_conditional_losses_444352"
 conv2d_4/StatefulPartitionedCallУ
max_pooling2d_4/PartitionedCallPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         39* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_439882!
max_pooling2d_4/PartitionedCall╕
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_4/PartitionedCall:output:0batch_normalization_6_44898batch_normalization_6_44900batch_normalization_6_44902batch_normalization_6_44904*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         39*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Y
fTRR
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_444712/
-batch_normalization_6/StatefulPartitionedCall╔
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:0conv2d_5_44907conv2d_5_44909*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         39 *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_conv2d_5_layer_call_and_return_conditional_losses_445362"
 conv2d_5/StatefulPartitionedCallУ
max_pooling2d_5/PartitionedCallPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_441042!
max_pooling2d_5/PartitionedCall╕
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_5/PartitionedCall:output:0batch_normalization_7_44913batch_normalization_7_44915batch_normalization_7_44917batch_normalization_7_44919*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Y
fTRR
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_445722/
-batch_normalization_7/StatefulPartitionedCallЗ
flatten_1/PartitionedCallPartitionedCall6batch_normalization_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         рv* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_446322
flatten_1/PartitionedCallи
dense_3/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_3_44923dense_3_44925*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_446512!
dense_3/StatefulPartitionedCallР
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_446792#
!dropout_2/StatefulPartitionedCall░
dense_4/StatefulPartitionedCallStatefulPartitionedCall*dropout_2/StatefulPartitionedCall:output:0dense_4_44929dense_4_44931*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_447082!
dense_4/StatefulPartitionedCall┤
!dropout_3/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0"^dropout_2/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dropout_3_layer_call_and_return_conditional_losses_447362#
!dropout_3/StatefulPartitionedCall░
dense_5/StatefulPartitionedCallStatefulPartitionedCall*dropout_3/StatefulPartitionedCall:output:0dense_5_44935dense_5_44937*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dense_5_layer_call_and_return_conditional_losses_447652!
dense_5/StatefulPartitionedCall╙
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall.^batch_normalization_6/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall"^dropout_3/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*Ъ
_input_shapesИ
Е:         Ат	::::::::::::::::::::::::::::2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2^
-batch_normalization_6/StatefulPartitionedCall-batch_normalization_6/StatefulPartitionedCall2^
-batch_normalization_7/StatefulPartitionedCall-batch_normalization_7/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall2F
!dropout_3/StatefulPartitionedCall!dropout_3/StatefulPartitionedCall:Q M
)
_output_shapes
:         Ат	
 
_user_specified_nameinputs
╠
Ч
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_46070

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
FusedBatchNormV3н
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue╗
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1Р
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
ю	
█
B__inference_dense_5_layer_call_and_return_conditional_losses_46738

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:         2	
SigmoidР
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*.
_input_shapes
:         ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
╫
и
5__inference_batch_normalization_5_layer_call_fn_46313

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCallб
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         gл*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Y
fTRR
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_443702
StatefulPartitionedCallЧ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:         gл2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:         gл::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:         gл
 
_user_specified_nameinputs
└
є
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_46532

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1▄
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3ь
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                            ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
Э
и
5__inference_batch_normalization_4_layer_call_fn_46114

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCall┤
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Y
fTRR
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_438552
StatefulPartitionedCallи
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
╠
Ч
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_46514

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
FusedBatchNormV3н
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue╗
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1Р
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                            ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
■
}
(__inference_conv2d_4_layer_call_fn_46346

inputs
unknown
	unknown_0
identityИвStatefulPartitionedCall№
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         gл*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_conv2d_4_layer_call_and_return_conditional_losses_444352
StatefulPartitionedCallЧ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:         gл2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:         gл::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:         gл
 
_user_specified_nameinputs
ь	
█
B__inference_dense_4_layer_call_and_return_conditional_losses_46691

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:         2
ReluЧ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*.
_input_shapes
:         @::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
А
є
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_46152

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1╠
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:         ╖Б:::::*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3▄
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*1
_output_shapes
:         ╖Б2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:         ╖Б::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:         ╖Б
 
_user_specified_nameinputs
л
K
/__inference_max_pooling2d_4_layer_call_fn_43994

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
6:4                                    * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_439882
PartitionedCallП
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
Ы
и
5__inference_batch_normalization_7_layer_call_fn_46545

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCall▓
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Y
fTRR
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_441722
StatefulPartitionedCallи
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                            ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
ь	
█
B__inference_dense_4_layer_call_and_return_conditional_losses_44708

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:         2
ReluЧ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*.
_input_shapes
:         @::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
╪
|
'__inference_dense_4_layer_call_fn_46700

inputs
unknown
	unknown_0
identityИвStatefulPartitionedCallЄ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_447082
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*.
_input_shapes
:         @::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
у╡
Ш
 __inference__wrapped_model_43598
reshape_1_input>
:sequential_1_batch_normalization_4_readvariableop_resource@
<sequential_1_batch_normalization_4_readvariableop_1_resourceO
Ksequential_1_batch_normalization_4_fusedbatchnormv3_readvariableop_resourceQ
Msequential_1_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource8
4sequential_1_conv2d_3_conv2d_readvariableop_resource9
5sequential_1_conv2d_3_biasadd_readvariableop_resource>
:sequential_1_batch_normalization_5_readvariableop_resource@
<sequential_1_batch_normalization_5_readvariableop_1_resourceO
Ksequential_1_batch_normalization_5_fusedbatchnormv3_readvariableop_resourceQ
Msequential_1_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource8
4sequential_1_conv2d_4_conv2d_readvariableop_resource9
5sequential_1_conv2d_4_biasadd_readvariableop_resource>
:sequential_1_batch_normalization_6_readvariableop_resource@
<sequential_1_batch_normalization_6_readvariableop_1_resourceO
Ksequential_1_batch_normalization_6_fusedbatchnormv3_readvariableop_resourceQ
Msequential_1_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource8
4sequential_1_conv2d_5_conv2d_readvariableop_resource9
5sequential_1_conv2d_5_biasadd_readvariableop_resource>
:sequential_1_batch_normalization_7_readvariableop_resource@
<sequential_1_batch_normalization_7_readvariableop_1_resourceO
Ksequential_1_batch_normalization_7_fusedbatchnormv3_readvariableop_resourceQ
Msequential_1_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource7
3sequential_1_dense_3_matmul_readvariableop_resource8
4sequential_1_dense_3_biasadd_readvariableop_resource7
3sequential_1_dense_4_matmul_readvariableop_resource8
4sequential_1_dense_4_biasadd_readvariableop_resource7
3sequential_1_dense_5_matmul_readvariableop_resource8
4sequential_1_dense_5_biasadd_readvariableop_resource
identityИвBsequential_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOpвDsequential_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1в1sequential_1/batch_normalization_4/ReadVariableOpв3sequential_1/batch_normalization_4/ReadVariableOp_1вBsequential_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOpвDsequential_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1в1sequential_1/batch_normalization_5/ReadVariableOpв3sequential_1/batch_normalization_5/ReadVariableOp_1вBsequential_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOpвDsequential_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1в1sequential_1/batch_normalization_6/ReadVariableOpв3sequential_1/batch_normalization_6/ReadVariableOp_1вBsequential_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOpвDsequential_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1в1sequential_1/batch_normalization_7/ReadVariableOpв3sequential_1/batch_normalization_7/ReadVariableOp_1в,sequential_1/conv2d_3/BiasAdd/ReadVariableOpв+sequential_1/conv2d_3/Conv2D/ReadVariableOpв,sequential_1/conv2d_4/BiasAdd/ReadVariableOpв+sequential_1/conv2d_4/Conv2D/ReadVariableOpв,sequential_1/conv2d_5/BiasAdd/ReadVariableOpв+sequential_1/conv2d_5/Conv2D/ReadVariableOpв+sequential_1/dense_3/BiasAdd/ReadVariableOpв*sequential_1/dense_3/MatMul/ReadVariableOpв+sequential_1/dense_4/BiasAdd/ReadVariableOpв*sequential_1/dense_4/MatMul/ReadVariableOpв+sequential_1/dense_5/BiasAdd/ReadVariableOpв*sequential_1/dense_5/MatMul/ReadVariableOp{
sequential_1/reshape_1/ShapeShapereshape_1_input*
T0*
_output_shapes
:2
sequential_1/reshape_1/Shapeв
*sequential_1/reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*sequential_1/reshape_1/strided_slice/stackж
,sequential_1/reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_1/reshape_1/strided_slice/stack_1ж
,sequential_1/reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_1/reshape_1/strided_slice/stack_2ь
$sequential_1/reshape_1/strided_sliceStridedSlice%sequential_1/reshape_1/Shape:output:03sequential_1/reshape_1/strided_slice/stack:output:05sequential_1/reshape_1/strided_slice/stack_1:output:05sequential_1/reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$sequential_1/reshape_1/strided_sliceФ
&sequential_1/reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB	 :Ат	2(
&sequential_1/reshape_1/Reshape/shape/1Т
&sequential_1/reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2(
&sequential_1/reshape_1/Reshape/shape/2У
$sequential_1/reshape_1/Reshape/shapePack-sequential_1/reshape_1/strided_slice:output:0/sequential_1/reshape_1/Reshape/shape/1:output:0/sequential_1/reshape_1/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2&
$sequential_1/reshape_1/Reshape/shape├
sequential_1/reshape_1/ReshapeReshapereshape_1_input-sequential_1/reshape_1/Reshape/shape:output:0*
T0*-
_output_shapes
:         Ат	2 
sequential_1/reshape_1/Reshape╗
1sequential_1/stft_magnitude/stft_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          23
1sequential_1/stft_magnitude/stft_1/transpose/permЖ
,sequential_1/stft_magnitude/stft_1/transpose	Transpose'sequential_1/reshape_1/Reshape:output:0:sequential_1/stft_magnitude/stft_1/transpose/perm:output:0*
T0*-
_output_shapes
:         Ат	2.
,sequential_1/stft_magnitude/stft_1/transpose╤
Esequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame_lengthConst*
_output_shapes
: *
dtype0*
value
B :А2G
Esequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame_length═
Csequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame_stepConst*
_output_shapes
: *
dtype0*
value
B :А2E
Csequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame_step═
Csequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/fft_lengthConst*
_output_shapes
: *
dtype0*
value
B :А2E
Csequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/fft_length╒
Csequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/axisConst*
_output_shapes
: *
dtype0*
valueB :
         2E
Csequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/axisь
Dsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/ShapeShape0sequential_1/stft_magnitude/stft_1/transpose:y:0*
T0*
_output_shapes
:2F
Dsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/Shape╠
Csequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/RankConst*
_output_shapes
: *
dtype0*
value	B :2E
Csequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/Rank┌
Jsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2L
Jsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/range/start┌
Jsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2L
Jsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/range/deltaй
Dsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/rangeRangeSsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/range/start:output:0Lsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/Rank:output:0Ssequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/range/delta:output:0*
_output_shapes
:2F
Dsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/range√
Rsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2T
Rsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/strided_slice/stackЎ
Tsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2V
Tsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/strided_slice/stack_1Ў
Tsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2V
Tsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/strided_slice/stack_2▄
Lsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/strided_sliceStridedSliceMsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/range:output:0[sequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/strided_slice/stack:output:0]sequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/strided_slice/stack_1:output:0]sequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2N
Lsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/strided_slice╬
Dsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/sub/yConst*
_output_shapes
: *
dtype0*
value	B :2F
Dsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/sub/y═
Bsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/subSubLsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/Rank:output:0Msequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/sub/y:output:0*
T0*
_output_shapes
: 2D
Bsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/sub╙
Dsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/sub_1SubFsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/sub:z:0Usequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/strided_slice:output:0*
T0*
_output_shapes
: 2F
Dsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/sub_1╘
Gsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2I
Gsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/packed/1╖
Esequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/packedPackUsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/strided_slice:output:0Psequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/packed/1:output:0Hsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/sub_1:z:0*
N*
T0*
_output_shapes
:2G
Esequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/packedт
Nsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2P
Nsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/split/split_dim┌
Dsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/splitSplitVMsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/Shape:output:0Nsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/packed:output:0Wsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/split/split_dim:output:0*
T0*

Tlen0*$
_output_shapes
::: *
	num_split2F
Dsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/split▀
Lsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/Reshape/shapeConst*
_output_shapes
: *
dtype0*
valueB 2N
Lsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/Reshape/shapeу
Nsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/Reshape/shape_1Const*
_output_shapes
: *
dtype0*
valueB 2P
Nsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/Reshape/shape_1ф
Fsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/ReshapeReshapeMsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/split:output:1Wsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/Reshape/shape_1:output:0*
T0*
_output_shapes
: 2H
Fsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/Reshape╠
Csequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/SizeConst*
_output_shapes
: *
dtype0*
value	B :2E
Csequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/Size╨
Esequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/Size_1Const*
_output_shapes
: *
dtype0*
value	B : 2G
Esequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/Size_1╒
Dsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/sub_2SubOsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/Reshape:output:0Nsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame_length:output:0*
T0*
_output_shapes
: 2F
Dsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/sub_2╫
Gsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/floordivFloorDivHsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/sub_2:z:0Lsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame_step:output:0*
T0*
_output_shapes
: 2I
Gsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/floordiv╬
Dsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/add/xConst*
_output_shapes
: *
dtype0*
value	B :2F
Dsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/add/x╬
Bsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/addAddV2Msequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/add/x:output:0Ksequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/floordiv:z:0*
T0*
_output_shapes
: 2D
Bsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/add╓
Hsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/Maximum/xConst*
_output_shapes
: *
dtype0*
value	B : 2J
Hsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/Maximum/x╫
Fsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/MaximumMaximumQsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/Maximum/x:output:0Fsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/add:z:0*
T0*
_output_shapes
: 2H
Fsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/Maximum╫
Hsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/gcd/ConstConst*
_output_shapes
: *
dtype0*
value
B :А2J
Hsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/gcd/Const▌
Ksequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/floordiv_1/yConst*
_output_shapes
: *
dtype0*
value
B :А2M
Ksequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/floordiv_1/yщ
Isequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/floordiv_1FloorDivNsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame_length:output:0Tsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/floordiv_1/y:output:0*
T0*
_output_shapes
: 2K
Isequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/floordiv_1▌
Ksequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/floordiv_2/yConst*
_output_shapes
: *
dtype0*
value
B :А2M
Ksequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/floordiv_2/yч
Isequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/floordiv_2FloorDivLsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame_step:output:0Tsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/floordiv_2/y:output:0*
T0*
_output_shapes
: 2K
Isequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/floordiv_2▌
Ksequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/floordiv_3/yConst*
_output_shapes
: *
dtype0*
value
B :А2M
Ksequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/floordiv_3/yъ
Isequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/floordiv_3FloorDivOsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/Reshape:output:0Tsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/floordiv_3/y:output:0*
T0*
_output_shapes
: 2K
Isequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/floordiv_3╧
Dsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/mul/yConst*
_output_shapes
: *
dtype0*
value
B :А2F
Dsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/mul/y╬
Bsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/mulMulMsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/floordiv_3:z:0Msequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/mul/y:output:0*
T0*
_output_shapes
: 2D
Bsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/mulЮ
Nsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/concat/values_1PackFsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/mul:z:0*
N*
T0*
_output_shapes
:2P
Nsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/concat/values_1┌
Jsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2L
Jsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/concat/axisФ
Esequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/concatConcatV2Msequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/split:output:0Wsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/concat/values_1:output:0Msequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/split:output:2Ssequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/concat/axis:output:0*
N*
T0*
_output_shapes
:2G
Esequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/concatы
Rsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/concat_1/values_1/1Const*
_output_shapes
: *
dtype0*
value
B :А2T
Rsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/concat_1/values_1/1Ж
Psequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/concat_1/values_1PackMsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/floordiv_3:z:0[sequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/concat_1/values_1/1:output:0*
N*
T0*
_output_shapes
:2R
Psequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/concat_1/values_1▐
Lsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2N
Lsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/concat_1/axisЬ
Gsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/concat_1ConcatV2Msequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/split:output:0Ysequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/concat_1/values_1:output:0Msequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/split:output:2Usequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2I
Gsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/concat_1р
Isequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/zeros_likeConst*
_output_shapes
:*
dtype0*
valueB: 2K
Isequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/zeros_likeъ
Nsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/ones_like/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2P
Nsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/ones_like/Shapeт
Nsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/ones_like/ConstConst*
_output_shapes
: *
dtype0*
value	B :2P
Nsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/ones_like/Constє
Hsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/ones_likeFillWsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/ones_like/Shape:output:0Wsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/ones_like/Const:output:0*
T0*
_output_shapes
:2J
Hsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/ones_likeи
Ksequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/StridedSliceStridedSlice0sequential_1/stft_magnitude/stft_1/transpose:y:0Rsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/zeros_like:output:0Nsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/concat:output:0Qsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/ones_like:output:0*
Index0*
T0*=
_output_shapes+
):'                           2M
Ksequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/StridedSliceФ
Hsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/Reshape_1ReshapeTsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/StridedSlice:output:0Psequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/concat_1:output:0*
T0*B
_output_shapes0
.:,                           А2J
Hsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/Reshape_1▐
Lsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/range_1/startConst*
_output_shapes
: *
dtype0*
value	B : 2N
Lsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/range_1/start▐
Lsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :2N
Lsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/range_1/delta╕
Fsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/range_1RangeUsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/range_1/start:output:0Jsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/Maximum:z:0Usequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/range_1/delta:output:0*#
_output_shapes
:         2H
Fsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/range_1с
Dsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/mul_1MulOsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/range_1:output:0Msequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/floordiv_2:z:0*
T0*#
_output_shapes
:         2F
Dsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/mul_1ц
Psequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2R
Psequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/Reshape_2/shape/1¤
Nsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/Reshape_2/shapePackJsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/Maximum:z:0Ysequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/Reshape_2/shape/1:output:0*
N*
T0*
_output_shapes
:2P
Nsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/Reshape_2/shapeЇ
Hsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/Reshape_2ReshapeHsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/mul_1:z:0Wsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/Reshape_2/shape:output:0*
T0*'
_output_shapes
:         2J
Hsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/Reshape_2▐
Lsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/range_2/startConst*
_output_shapes
: *
dtype0*
value	B : 2N
Lsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/range_2/start▐
Lsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/range_2/deltaConst*
_output_shapes
: *
dtype0*
value	B :2N
Lsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/range_2/delta▓
Fsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/range_2RangeUsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/range_2/start:output:0Msequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/floordiv_1:z:0Usequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/range_2/delta:output:0*
_output_shapes
:2H
Fsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/range_2ц
Psequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/Reshape_3/shape/0Const*
_output_shapes
: *
dtype0*
value	B :2R
Psequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/Reshape_3/shape/0А
Nsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/Reshape_3/shapePackYsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/Reshape_3/shape/0:output:0Msequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/floordiv_1:z:0*
N*
T0*
_output_shapes
:2P
Nsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/Reshape_3/shapeЄ
Hsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/Reshape_3ReshapeOsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/range_2:output:0Wsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/Reshape_3/shape:output:0*
T0*
_output_shapes

:2J
Hsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/Reshape_3э
Dsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/add_1AddV2Qsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/Reshape_2:output:0Qsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/Reshape_3:output:0*
T0*'
_output_shapes
:         2F
Dsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/add_1Ж
Gsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/GatherV2GatherV2Qsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/Reshape_1:output:0Hsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/add_1:z:0Usequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/strided_slice:output:0*
Taxis0*
Tindices0*
Tparams0*F
_output_shapes4
2:0                           А2I
Gsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/GatherV2Ў
Psequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/concat_2/values_1PackJsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/Maximum:z:0Nsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame_length:output:0*
N*
T0*
_output_shapes
:2R
Psequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/concat_2/values_1▐
Lsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2N
Lsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/concat_2/axisЬ
Gsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/concat_2ConcatV2Msequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/split:output:0Ysequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/concat_2/values_1:output:0Msequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/split:output:2Usequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/concat_2/axis:output:0*
N*
T0*
_output_shapes
:2I
Gsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/concat_2 
Hsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/Reshape_4ReshapePsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/GatherV2:output:0Psequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/concat_2:output:0*
T0*1
_output_shapes
:         ╖А2J
Hsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/Reshape_4р
Msequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/periodicConst*
_output_shapes
: *
dtype0
*
value	B
 Z2O
Msequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/periodicж
Isequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/CastCastVsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/periodic:output:0*

DstT0*

SrcT0
*
_output_shapes
: 2K
Isequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/Castф
Osequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/FloorMod/yConst*
_output_shapes
: *
dtype0*
value	B :2Q
Osequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/FloorMod/yї
Msequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/FloorModFloorModNsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame_length:output:0Xsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/FloorMod/y:output:0*
T0*
_output_shapes
: 2O
Msequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/FloorMod┌
Jsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/sub/xConst*
_output_shapes
: *
dtype0*
value	B :2L
Jsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/sub/xф
Hsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/subSubSsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/sub/x:output:0Qsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/FloorMod:z:0*
T0*
_output_shapes
: 2J
Hsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/sub┘
Hsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/mulMulMsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/Cast:y:0Lsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/sub:z:0*
T0*
_output_shapes
: 2J
Hsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/mul▄
Hsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/addAddV2Nsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame_length:output:0Lsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/mul:z:0*
T0*
_output_shapes
: 2J
Hsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/add▐
Lsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/sub_1/yConst*
_output_shapes
: *
dtype0*
value	B :2N
Lsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/sub_1/yх
Jsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/sub_1SubLsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/add:z:0Usequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/sub_1/y:output:0*
T0*
_output_shapes
: 2L
Jsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/sub_1в
Ksequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/Cast_1CastNsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/sub_1:z:0*

DstT0*

SrcT0*
_output_shapes
: 2M
Ksequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/Cast_1ц
Psequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2R
Psequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/range/startц
Psequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2R
Psequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/range/delta─
Jsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/rangeRangeYsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/range/start:output:0Nsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame_length:output:0Ysequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/range/delta:output:0*
_output_shapes	
:А2L
Jsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/rangeм
Ksequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/Cast_2CastSsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/range:output:0*

DstT0*

SrcT0*
_output_shapes	
:А2M
Ksequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/Cast_2▌
Jsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *█╔@2L
Jsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/Constы
Jsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/mul_1MulSsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/Const:output:0Osequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/Cast_2:y:0*
T0*
_output_shapes	
:А2L
Jsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/mul_1ю
Lsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/truedivRealDivNsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/mul_1:z:0Osequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/Cast_1:y:0*
T0*
_output_shapes	
:А2N
Lsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/truedivУ
Hsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/CosCosPsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/truediv:z:0*
T0*
_output_shapes	
:А2J
Hsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/Cosс
Lsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2N
Lsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/mul_2/xъ
Jsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/mul_2MulUsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/mul_2/x:output:0Lsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/Cos:y:0*
T0*
_output_shapes	
:А2L
Jsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/mul_2с
Lsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2N
Lsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/sub_2/xь
Jsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/sub_2SubUsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/sub_2/x:output:0Nsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/mul_2:z:0*
T0*
_output_shapes	
:А2L
Jsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/sub_2т
<sequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/mulMulQsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/frame/Reshape_4:output:0Nsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/hann_window/sub_2:z:0*
T0*1
_output_shapes
:         ╖А2>
<sequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/mulР
Dsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/rfft/packedPackLsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/fft_length:output:0*
N*
T0*
_output_shapes
:2F
Dsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/rfft/packed▀
Hsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/rfft/fft_lengthConst*
_output_shapes
:*
dtype0*
valueB:А2J
Hsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/rfft/fft_length╬
=sequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/rfftRFFT@sequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/mul:z:0Qsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/rfft/fft_length:output:0*1
_output_shapes
:         ╖Б2?
=sequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/rfft├
3sequential_1/stft_magnitude/stft_1/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             25
3sequential_1/stft_magnitude/stft_1/transpose_1/permп
.sequential_1/stft_magnitude/stft_1/transpose_1	TransposeFsequential_1/stft_magnitude/stft_1/stft_1_tf.signal.stft/rfft:output:0<sequential_1/stft_magnitude/stft_1/transpose_1/perm:output:0*
T0*1
_output_shapes
:         ╖Б20
.sequential_1/stft_magnitude/stft_1/transpose_1╧
+sequential_1/stft_magnitude/magnitude_1/Abs
ComplexAbs2sequential_1/stft_magnitude/stft_1/transpose_1:y:0*1
_output_shapes
:         ╖Б2-
+sequential_1/stft_magnitude/magnitude_1/Abs▌
1sequential_1/batch_normalization_4/ReadVariableOpReadVariableOp:sequential_1_batch_normalization_4_readvariableop_resource*
_output_shapes
:*
dtype023
1sequential_1/batch_normalization_4/ReadVariableOpу
3sequential_1/batch_normalization_4/ReadVariableOp_1ReadVariableOp<sequential_1_batch_normalization_4_readvariableop_1_resource*
_output_shapes
:*
dtype025
3sequential_1/batch_normalization_4/ReadVariableOp_1Р
Bsequential_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_1_batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02D
Bsequential_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOpЦ
Dsequential_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_1_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02F
Dsequential_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1╟
3sequential_1/batch_normalization_4/FusedBatchNormV3FusedBatchNormV3/sequential_1/stft_magnitude/magnitude_1/Abs:y:09sequential_1/batch_normalization_4/ReadVariableOp:value:0;sequential_1/batch_normalization_4/ReadVariableOp_1:value:0Jsequential_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:         ╖Б:::::*
epsilon%oГ:*
is_training( 25
3sequential_1/batch_normalization_4/FusedBatchNormV3╫
+sequential_1/conv2d_3/Conv2D/ReadVariableOpReadVariableOp4sequential_1_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02-
+sequential_1/conv2d_3/Conv2D/ReadVariableOpШ
sequential_1/conv2d_3/Conv2DConv2D7sequential_1/batch_normalization_4/FusedBatchNormV3:y:03sequential_1/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ╖Б*
paddingSAME*
strides
2
sequential_1/conv2d_3/Conv2D╬
,sequential_1/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp5sequential_1_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,sequential_1/conv2d_3/BiasAdd/ReadVariableOpт
sequential_1/conv2d_3/BiasAddBiasAdd%sequential_1/conv2d_3/Conv2D:output:04sequential_1/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ╖Б2
sequential_1/conv2d_3/BiasAddд
sequential_1/conv2d_3/ReluRelu&sequential_1/conv2d_3/BiasAdd:output:0*
T0*1
_output_shapes
:         ╖Б2
sequential_1/conv2d_3/Reluя
$sequential_1/max_pooling2d_3/MaxPoolMaxPool(sequential_1/conv2d_3/Relu:activations:0*0
_output_shapes
:         gл*
ksize
*
paddingVALID*
strides
2&
$sequential_1/max_pooling2d_3/MaxPool▌
1sequential_1/batch_normalization_5/ReadVariableOpReadVariableOp:sequential_1_batch_normalization_5_readvariableop_resource*
_output_shapes
:*
dtype023
1sequential_1/batch_normalization_5/ReadVariableOpу
3sequential_1/batch_normalization_5/ReadVariableOp_1ReadVariableOp<sequential_1_batch_normalization_5_readvariableop_1_resource*
_output_shapes
:*
dtype025
3sequential_1/batch_normalization_5/ReadVariableOp_1Р
Bsequential_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_1_batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02D
Bsequential_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOpЦ
Dsequential_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_1_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02F
Dsequential_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1─
3sequential_1/batch_normalization_5/FusedBatchNormV3FusedBatchNormV3-sequential_1/max_pooling2d_3/MaxPool:output:09sequential_1/batch_normalization_5/ReadVariableOp:value:0;sequential_1/batch_normalization_5/ReadVariableOp_1:value:0Jsequential_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:         gл:::::*
epsilon%oГ:*
is_training( 25
3sequential_1/batch_normalization_5/FusedBatchNormV3╫
+sequential_1/conv2d_4/Conv2D/ReadVariableOpReadVariableOp4sequential_1_conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02-
+sequential_1/conv2d_4/Conv2D/ReadVariableOpЧ
sequential_1/conv2d_4/Conv2DConv2D7sequential_1/batch_normalization_5/FusedBatchNormV3:y:03sequential_1/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         gл*
paddingSAME*
strides
2
sequential_1/conv2d_4/Conv2D╬
,sequential_1/conv2d_4/BiasAdd/ReadVariableOpReadVariableOp5sequential_1_conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,sequential_1/conv2d_4/BiasAdd/ReadVariableOpс
sequential_1/conv2d_4/BiasAddBiasAdd%sequential_1/conv2d_4/Conv2D:output:04sequential_1/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         gл2
sequential_1/conv2d_4/BiasAddг
sequential_1/conv2d_4/ReluRelu&sequential_1/conv2d_4/BiasAdd:output:0*
T0*0
_output_shapes
:         gл2
sequential_1/conv2d_4/Reluю
$sequential_1/max_pooling2d_4/MaxPoolMaxPool(sequential_1/conv2d_4/Relu:activations:0*/
_output_shapes
:         39*
ksize
*
paddingVALID*
strides
2&
$sequential_1/max_pooling2d_4/MaxPool▌
1sequential_1/batch_normalization_6/ReadVariableOpReadVariableOp:sequential_1_batch_normalization_6_readvariableop_resource*
_output_shapes
:*
dtype023
1sequential_1/batch_normalization_6/ReadVariableOpу
3sequential_1/batch_normalization_6/ReadVariableOp_1ReadVariableOp<sequential_1_batch_normalization_6_readvariableop_1_resource*
_output_shapes
:*
dtype025
3sequential_1/batch_normalization_6/ReadVariableOp_1Р
Bsequential_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_1_batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02D
Bsequential_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOpЦ
Dsequential_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_1_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02F
Dsequential_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1├
3sequential_1/batch_normalization_6/FusedBatchNormV3FusedBatchNormV3-sequential_1/max_pooling2d_4/MaxPool:output:09sequential_1/batch_normalization_6/ReadVariableOp:value:0;sequential_1/batch_normalization_6/ReadVariableOp_1:value:0Jsequential_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         39:::::*
epsilon%oГ:*
is_training( 25
3sequential_1/batch_normalization_6/FusedBatchNormV3╫
+sequential_1/conv2d_5/Conv2D/ReadVariableOpReadVariableOp4sequential_1_conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02-
+sequential_1/conv2d_5/Conv2D/ReadVariableOpЦ
sequential_1/conv2d_5/Conv2DConv2D7sequential_1/batch_normalization_6/FusedBatchNormV3:y:03sequential_1/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         39 *
paddingSAME*
strides
2
sequential_1/conv2d_5/Conv2D╬
,sequential_1/conv2d_5/BiasAdd/ReadVariableOpReadVariableOp5sequential_1_conv2d_5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,sequential_1/conv2d_5/BiasAdd/ReadVariableOpр
sequential_1/conv2d_5/BiasAddBiasAdd%sequential_1/conv2d_5/Conv2D:output:04sequential_1/conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         39 2
sequential_1/conv2d_5/BiasAddв
sequential_1/conv2d_5/ReluRelu&sequential_1/conv2d_5/BiasAdd:output:0*
T0*/
_output_shapes
:         39 2
sequential_1/conv2d_5/Reluю
$sequential_1/max_pooling2d_5/MaxPoolMaxPool(sequential_1/conv2d_5/Relu:activations:0*/
_output_shapes
:          *
ksize
*
paddingVALID*
strides
2&
$sequential_1/max_pooling2d_5/MaxPool▌
1sequential_1/batch_normalization_7/ReadVariableOpReadVariableOp:sequential_1_batch_normalization_7_readvariableop_resource*
_output_shapes
: *
dtype023
1sequential_1/batch_normalization_7/ReadVariableOpу
3sequential_1/batch_normalization_7/ReadVariableOp_1ReadVariableOp<sequential_1_batch_normalization_7_readvariableop_1_resource*
_output_shapes
: *
dtype025
3sequential_1/batch_normalization_7/ReadVariableOp_1Р
Bsequential_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_1_batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02D
Bsequential_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOpЦ
Dsequential_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_1_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02F
Dsequential_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1├
3sequential_1/batch_normalization_7/FusedBatchNormV3FusedBatchNormV3-sequential_1/max_pooling2d_5/MaxPool:output:09sequential_1/batch_normalization_7/ReadVariableOp:value:0;sequential_1/batch_normalization_7/ReadVariableOp_1:value:0Jsequential_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:          : : : : :*
epsilon%oГ:*
is_training( 25
3sequential_1/batch_normalization_7/FusedBatchNormV3Н
sequential_1/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"    `;  2
sequential_1/flatten_1/Const▐
sequential_1/flatten_1/ReshapeReshape7sequential_1/batch_normalization_7/FusedBatchNormV3:y:0%sequential_1/flatten_1/Const:output:0*
T0*(
_output_shapes
:         рv2 
sequential_1/flatten_1/Reshape═
*sequential_1/dense_3/MatMul/ReadVariableOpReadVariableOp3sequential_1_dense_3_matmul_readvariableop_resource*
_output_shapes
:	рv@*
dtype02,
*sequential_1/dense_3/MatMul/ReadVariableOp╙
sequential_1/dense_3/MatMulMatMul'sequential_1/flatten_1/Reshape:output:02sequential_1/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2
sequential_1/dense_3/MatMul╦
+sequential_1/dense_3/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02-
+sequential_1/dense_3/BiasAdd/ReadVariableOp╒
sequential_1/dense_3/BiasAddBiasAdd%sequential_1/dense_3/MatMul:product:03sequential_1/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2
sequential_1/dense_3/BiasAddЧ
sequential_1/dense_3/ReluRelu%sequential_1/dense_3/BiasAdd:output:0*
T0*'
_output_shapes
:         @2
sequential_1/dense_3/Reluй
sequential_1/dropout_2/IdentityIdentity'sequential_1/dense_3/Relu:activations:0*
T0*'
_output_shapes
:         @2!
sequential_1/dropout_2/Identity╠
*sequential_1/dense_4/MatMul/ReadVariableOpReadVariableOp3sequential_1_dense_4_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02,
*sequential_1/dense_4/MatMul/ReadVariableOp╘
sequential_1/dense_4/MatMulMatMul(sequential_1/dropout_2/Identity:output:02sequential_1/dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
sequential_1/dense_4/MatMul╦
+sequential_1/dense_4/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+sequential_1/dense_4/BiasAdd/ReadVariableOp╒
sequential_1/dense_4/BiasAddBiasAdd%sequential_1/dense_4/MatMul:product:03sequential_1/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
sequential_1/dense_4/BiasAddЧ
sequential_1/dense_4/ReluRelu%sequential_1/dense_4/BiasAdd:output:0*
T0*'
_output_shapes
:         2
sequential_1/dense_4/Reluй
sequential_1/dropout_3/IdentityIdentity'sequential_1/dense_4/Relu:activations:0*
T0*'
_output_shapes
:         2!
sequential_1/dropout_3/Identity╠
*sequential_1/dense_5/MatMul/ReadVariableOpReadVariableOp3sequential_1_dense_5_matmul_readvariableop_resource*
_output_shapes

:*
dtype02,
*sequential_1/dense_5/MatMul/ReadVariableOp╘
sequential_1/dense_5/MatMulMatMul(sequential_1/dropout_3/Identity:output:02sequential_1/dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
sequential_1/dense_5/MatMul╦
+sequential_1/dense_5/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+sequential_1/dense_5/BiasAdd/ReadVariableOp╒
sequential_1/dense_5/BiasAddBiasAdd%sequential_1/dense_5/MatMul:product:03sequential_1/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
sequential_1/dense_5/BiasAddа
sequential_1/dense_5/SigmoidSigmoid%sequential_1/dense_5/BiasAdd:output:0*
T0*'
_output_shapes
:         2
sequential_1/dense_5/SigmoidЇ
IdentityIdentity sequential_1/dense_5/Sigmoid:y:0C^sequential_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOpE^sequential_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12^sequential_1/batch_normalization_4/ReadVariableOp4^sequential_1/batch_normalization_4/ReadVariableOp_1C^sequential_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOpE^sequential_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_12^sequential_1/batch_normalization_5/ReadVariableOp4^sequential_1/batch_normalization_5/ReadVariableOp_1C^sequential_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOpE^sequential_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_12^sequential_1/batch_normalization_6/ReadVariableOp4^sequential_1/batch_normalization_6/ReadVariableOp_1C^sequential_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOpE^sequential_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_12^sequential_1/batch_normalization_7/ReadVariableOp4^sequential_1/batch_normalization_7/ReadVariableOp_1-^sequential_1/conv2d_3/BiasAdd/ReadVariableOp,^sequential_1/conv2d_3/Conv2D/ReadVariableOp-^sequential_1/conv2d_4/BiasAdd/ReadVariableOp,^sequential_1/conv2d_4/Conv2D/ReadVariableOp-^sequential_1/conv2d_5/BiasAdd/ReadVariableOp,^sequential_1/conv2d_5/Conv2D/ReadVariableOp,^sequential_1/dense_3/BiasAdd/ReadVariableOp+^sequential_1/dense_3/MatMul/ReadVariableOp,^sequential_1/dense_4/BiasAdd/ReadVariableOp+^sequential_1/dense_4/MatMul/ReadVariableOp,^sequential_1/dense_5/BiasAdd/ReadVariableOp+^sequential_1/dense_5/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*Ъ
_input_shapesИ
Е:         Ат	::::::::::::::::::::::::::::2И
Bsequential_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOpBsequential_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp2М
Dsequential_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1Dsequential_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12f
1sequential_1/batch_normalization_4/ReadVariableOp1sequential_1/batch_normalization_4/ReadVariableOp2j
3sequential_1/batch_normalization_4/ReadVariableOp_13sequential_1/batch_normalization_4/ReadVariableOp_12И
Bsequential_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOpBsequential_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp2М
Dsequential_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1Dsequential_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_12f
1sequential_1/batch_normalization_5/ReadVariableOp1sequential_1/batch_normalization_5/ReadVariableOp2j
3sequential_1/batch_normalization_5/ReadVariableOp_13sequential_1/batch_normalization_5/ReadVariableOp_12И
Bsequential_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOpBsequential_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp2М
Dsequential_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1Dsequential_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_12f
1sequential_1/batch_normalization_6/ReadVariableOp1sequential_1/batch_normalization_6/ReadVariableOp2j
3sequential_1/batch_normalization_6/ReadVariableOp_13sequential_1/batch_normalization_6/ReadVariableOp_12И
Bsequential_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOpBsequential_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOp2М
Dsequential_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1Dsequential_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_12f
1sequential_1/batch_normalization_7/ReadVariableOp1sequential_1/batch_normalization_7/ReadVariableOp2j
3sequential_1/batch_normalization_7/ReadVariableOp_13sequential_1/batch_normalization_7/ReadVariableOp_12\
,sequential_1/conv2d_3/BiasAdd/ReadVariableOp,sequential_1/conv2d_3/BiasAdd/ReadVariableOp2Z
+sequential_1/conv2d_3/Conv2D/ReadVariableOp+sequential_1/conv2d_3/Conv2D/ReadVariableOp2\
,sequential_1/conv2d_4/BiasAdd/ReadVariableOp,sequential_1/conv2d_4/BiasAdd/ReadVariableOp2Z
+sequential_1/conv2d_4/Conv2D/ReadVariableOp+sequential_1/conv2d_4/Conv2D/ReadVariableOp2\
,sequential_1/conv2d_5/BiasAdd/ReadVariableOp,sequential_1/conv2d_5/BiasAdd/ReadVariableOp2Z
+sequential_1/conv2d_5/Conv2D/ReadVariableOp+sequential_1/conv2d_5/Conv2D/ReadVariableOp2Z
+sequential_1/dense_3/BiasAdd/ReadVariableOp+sequential_1/dense_3/BiasAdd/ReadVariableOp2X
*sequential_1/dense_3/MatMul/ReadVariableOp*sequential_1/dense_3/MatMul/ReadVariableOp2Z
+sequential_1/dense_4/BiasAdd/ReadVariableOp+sequential_1/dense_4/BiasAdd/ReadVariableOp2X
*sequential_1/dense_4/MatMul/ReadVariableOp*sequential_1/dense_4/MatMul/ReadVariableOp2Z
+sequential_1/dense_5/BiasAdd/ReadVariableOp+sequential_1/dense_5/BiasAdd/ReadVariableOp2X
*sequential_1/dense_5/MatMul/ReadVariableOp*sequential_1/dense_5/MatMul/ReadVariableOp:Z V
)
_output_shapes
:         Ат	
)
_user_specified_namereshape_1_input
═

▄
C__inference_conv2d_5_layer_call_and_return_conditional_losses_44536

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOpг
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         39 *
paddingSAME*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpИ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         39 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:         39 2
ReluЯ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:         39 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:         39::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         39
 
_user_specified_nameinputs
М
Ч
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_44269

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1┌
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:         ╖Б:::::*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
FusedBatchNormV3н
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue╗
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1А
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*1
_output_shapes
:         ╖Б2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:         ╖Б::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:         ╖Б
 
_user_specified_nameinputs
╓U
°	
G__inference_sequential_1_layer_call_and_return_conditional_losses_44860
reshape_1_input
batch_normalization_4_44787
batch_normalization_4_44789
batch_normalization_4_44791
batch_normalization_4_44793
conv2d_3_44796
conv2d_3_44798
batch_normalization_5_44802
batch_normalization_5_44804
batch_normalization_5_44806
batch_normalization_5_44808
conv2d_4_44811
conv2d_4_44813
batch_normalization_6_44817
batch_normalization_6_44819
batch_normalization_6_44821
batch_normalization_6_44823
conv2d_5_44826
conv2d_5_44828
batch_normalization_7_44832
batch_normalization_7_44834
batch_normalization_7_44836
batch_normalization_7_44838
dense_3_44842
dense_3_44844
dense_4_44848
dense_4_44850
dense_5_44854
dense_5_44856
identityИв-batch_normalization_4/StatefulPartitionedCallв-batch_normalization_5/StatefulPartitionedCallв-batch_normalization_6/StatefulPartitionedCallв-batch_normalization_7/StatefulPartitionedCallв conv2d_3/StatefulPartitionedCallв conv2d_4/StatefulPartitionedCallв conv2d_5/StatefulPartitionedCallвdense_3/StatefulPartitionedCallвdense_4/StatefulPartitionedCallвdense_5/StatefulPartitionedCallх
reshape_1/PartitionedCallPartitionedCallreshape_1_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:         Ат	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_reshape_1_layer_call_and_return_conditional_losses_442312
reshape_1/PartitionedCallЛ
stft_magnitude/PartitionedCallPartitionedCall"reshape_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ╖Б* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_stft_magnitude_layer_call_and_return_conditional_losses_437592 
stft_magnitude/PartitionedCall╗
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall'stft_magnitude/PartitionedCall:output:0batch_normalization_4_44787batch_normalization_4_44789batch_normalization_4_44791batch_normalization_4_44793*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ╖Б*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Y
fTRR
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_442872/
-batch_normalization_4/StatefulPartitionedCall╦
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0conv2d_3_44796conv2d_3_44798*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ╖Б*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_conv2d_3_layer_call_and_return_conditional_losses_443342"
 conv2d_3/StatefulPartitionedCallФ
max_pooling2d_3/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         gл* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_438722!
max_pooling2d_3/PartitionedCall╗
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_3/PartitionedCall:output:0batch_normalization_5_44802batch_normalization_5_44804batch_normalization_5_44806batch_normalization_5_44808*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         gл*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Y
fTRR
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_443882/
-batch_normalization_5/StatefulPartitionedCall╩
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0conv2d_4_44811conv2d_4_44813*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         gл*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_conv2d_4_layer_call_and_return_conditional_losses_444352"
 conv2d_4/StatefulPartitionedCallУ
max_pooling2d_4/PartitionedCallPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         39* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_439882!
max_pooling2d_4/PartitionedCall║
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_4/PartitionedCall:output:0batch_normalization_6_44817batch_normalization_6_44819batch_normalization_6_44821batch_normalization_6_44823*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         39*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Y
fTRR
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_444892/
-batch_normalization_6/StatefulPartitionedCall╔
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:0conv2d_5_44826conv2d_5_44828*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         39 *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_conv2d_5_layer_call_and_return_conditional_losses_445362"
 conv2d_5/StatefulPartitionedCallУ
max_pooling2d_5/PartitionedCallPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_441042!
max_pooling2d_5/PartitionedCall║
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_5/PartitionedCall:output:0batch_normalization_7_44832batch_normalization_7_44834batch_normalization_7_44836batch_normalization_7_44838*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Y
fTRR
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_445902/
-batch_normalization_7/StatefulPartitionedCallЗ
flatten_1/PartitionedCallPartitionedCall6batch_normalization_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         рv* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_446322
flatten_1/PartitionedCallи
dense_3/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_3_44842dense_3_44844*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_446512!
dense_3/StatefulPartitionedCall°
dropout_2/PartitionedCallPartitionedCall(dense_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_446842
dropout_2/PartitionedCallи
dense_4/StatefulPartitionedCallStatefulPartitionedCall"dropout_2/PartitionedCall:output:0dense_4_44848dense_4_44850*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_447082!
dense_4/StatefulPartitionedCall°
dropout_3/PartitionedCallPartitionedCall(dense_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dropout_3_layer_call_and_return_conditional_losses_447412
dropout_3/PartitionedCallи
dense_5/StatefulPartitionedCallStatefulPartitionedCall"dropout_3/PartitionedCall:output:0dense_5_44854dense_5_44856*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dense_5_layer_call_and_return_conditional_losses_447652!
dense_5/StatefulPartitionedCallЛ
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall.^batch_normalization_6/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*Ъ
_input_shapesИ
Е:         Ат	::::::::::::::::::::::::::::2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2^
-batch_normalization_6/StatefulPartitionedCall-batch_normalization_6/StatefulPartitionedCall2^
-batch_normalization_7/StatefulPartitionedCall-batch_normalization_7/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall:Z V
)
_output_shapes
:         Ат	
)
_user_specified_namereshape_1_input
╬
P
.__inference_stft_magnitude_layer_call_fn_43751
stft_1_input
identity╫
PartitionedCallPartitionedCallstft_1_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ╖Б* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_stft_magnitude_layer_call_and_return_conditional_losses_437482
PartitionedCallv
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:         ╖Б2

Identity"
identityIdentity:output:0*,
_input_shapes
:         Ат	:[ W
-
_output_shapes
:         Ат	
&
_user_specified_namestft_1_input
╠
Ч
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_46366

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
FusedBatchNormV3н
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue╗
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1Р
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
╜
]
F__inference_magnitude_1_layer_call_and_return_conditional_losses_43724
x
identityN
Abs
ComplexAbsx*1
_output_shapes
:         ╖Б2
Abse
IdentityIdentityAbs:y:0*
T0*1
_output_shapes
:         ╖Б2

Identity"
identityIdentity:output:0*0
_input_shapes
:         ╖Б:T P
1
_output_shapes
:         ╖Б

_user_specified_namex
И
Ч
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_44370

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1┘
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:         gл:::::*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
FusedBatchNormV3н
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue╗
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1 
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:         gл2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:         gл::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:         gл
 
_user_specified_nameinputs
─
Ш
,__inference_sequential_1_layer_call_fn_45741

inputs
unknown
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

unknown_26
identityИвStatefulPartitionedCall╥
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
unknown_26*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *6
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_449412
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*Ъ
_input_shapesИ
Е:         Ат	::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
)
_output_shapes
:         Ат	
 
_user_specified_nameinputs
М
Ч
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_46134

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1┌
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:         ╖Б:::::*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
FusedBatchNormV3н
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue╗
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1А
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*1
_output_shapes
:         ╖Б2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:         ╖Б::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:         ╖Б
 
_user_specified_nameinputs
└
є
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_43971

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1▄
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3ь
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
╠
Ч
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_43824

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
FusedBatchNormV3н
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue╗
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1Р
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
┘
и
5__inference_batch_normalization_5_layer_call_fn_46326

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCallг
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         gл*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Y
fTRR
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_443882
StatefulPartitionedCallЧ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:         gл2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:         gл::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:         gл
 
_user_specified_nameinputs
в
E
)__inference_reshape_1_layer_call_fn_45820

inputs
identity╚
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:         Ат	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_reshape_1_layer_call_and_return_conditional_losses_442312
PartitionedCallr
IdentityIdentityPartitionedCall:output:0*
T0*-
_output_shapes
:         Ат	2

Identity"
identityIdentity:output:0*(
_input_shapes
:         Ат	:Q M
)
_output_shapes
:         Ат	
 
_user_specified_nameinputs
┌
|
'__inference_dense_3_layer_call_fn_46653

inputs
unknown
	unknown_0
identityИвStatefulPartitionedCallЄ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_446512
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         @2

Identity"
identityIdentity:output:0*/
_input_shapes
:         рv::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         рv
 
_user_specified_nameinputs
╨▓
e
I__inference_stft_magnitude_layer_call_and_return_conditional_losses_45930

inputs
identityГ
stft_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
stft_1/transpose/permС
stft_1/transpose	Transposeinputsstft_1/transpose/perm:output:0*
T0*-
_output_shapes
:         Ат	2
stft_1/transposeЩ
)stft_1/stft_1_tf.signal.stft/frame_lengthConst*
_output_shapes
: *
dtype0*
value
B :А2+
)stft_1/stft_1_tf.signal.stft/frame_lengthХ
'stft_1/stft_1_tf.signal.stft/frame_stepConst*
_output_shapes
: *
dtype0*
value
B :А2)
'stft_1/stft_1_tf.signal.stft/frame_stepХ
'stft_1/stft_1_tf.signal.stft/fft_lengthConst*
_output_shapes
: *
dtype0*
value
B :А2)
'stft_1/stft_1_tf.signal.stft/fft_lengthЭ
'stft_1/stft_1_tf.signal.stft/frame/axisConst*
_output_shapes
: *
dtype0*
valueB :
         2)
'stft_1/stft_1_tf.signal.stft/frame/axisШ
(stft_1/stft_1_tf.signal.stft/frame/ShapeShapestft_1/transpose:y:0*
T0*
_output_shapes
:2*
(stft_1/stft_1_tf.signal.stft/frame/ShapeФ
'stft_1/stft_1_tf.signal.stft/frame/RankConst*
_output_shapes
: *
dtype0*
value	B :2)
'stft_1/stft_1_tf.signal.stft/frame/Rankв
.stft_1/stft_1_tf.signal.stft/frame/range/startConst*
_output_shapes
: *
dtype0*
value	B : 20
.stft_1/stft_1_tf.signal.stft/frame/range/startв
.stft_1/stft_1_tf.signal.stft/frame/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :20
.stft_1/stft_1_tf.signal.stft/frame/range/deltaЭ
(stft_1/stft_1_tf.signal.stft/frame/rangeRange7stft_1/stft_1_tf.signal.stft/frame/range/start:output:00stft_1/stft_1_tf.signal.stft/frame/Rank:output:07stft_1/stft_1_tf.signal.stft/frame/range/delta:output:0*
_output_shapes
:2*
(stft_1/stft_1_tf.signal.stft/frame/range├
6stft_1/stft_1_tf.signal.stft/frame/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
         28
6stft_1/stft_1_tf.signal.stft/frame/strided_slice/stack╛
8stft_1/stft_1_tf.signal.stft/frame/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2:
8stft_1/stft_1_tf.signal.stft/frame/strided_slice/stack_1╛
8stft_1/stft_1_tf.signal.stft/frame/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2:
8stft_1/stft_1_tf.signal.stft/frame/strided_slice/stack_2┤
0stft_1/stft_1_tf.signal.stft/frame/strided_sliceStridedSlice1stft_1/stft_1_tf.signal.stft/frame/range:output:0?stft_1/stft_1_tf.signal.stft/frame/strided_slice/stack:output:0Astft_1/stft_1_tf.signal.stft/frame/strided_slice/stack_1:output:0Astft_1/stft_1_tf.signal.stft/frame/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask22
0stft_1/stft_1_tf.signal.stft/frame/strided_sliceЦ
(stft_1/stft_1_tf.signal.stft/frame/sub/yConst*
_output_shapes
: *
dtype0*
value	B :2*
(stft_1/stft_1_tf.signal.stft/frame/sub/y▌
&stft_1/stft_1_tf.signal.stft/frame/subSub0stft_1/stft_1_tf.signal.stft/frame/Rank:output:01stft_1/stft_1_tf.signal.stft/frame/sub/y:output:0*
T0*
_output_shapes
: 2(
&stft_1/stft_1_tf.signal.stft/frame/subу
(stft_1/stft_1_tf.signal.stft/frame/sub_1Sub*stft_1/stft_1_tf.signal.stft/frame/sub:z:09stft_1/stft_1_tf.signal.stft/frame/strided_slice:output:0*
T0*
_output_shapes
: 2*
(stft_1/stft_1_tf.signal.stft/frame/sub_1Ь
+stft_1/stft_1_tf.signal.stft/frame/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2-
+stft_1/stft_1_tf.signal.stft/frame/packed/1л
)stft_1/stft_1_tf.signal.stft/frame/packedPack9stft_1/stft_1_tf.signal.stft/frame/strided_slice:output:04stft_1/stft_1_tf.signal.stft/frame/packed/1:output:0,stft_1/stft_1_tf.signal.stft/frame/sub_1:z:0*
N*
T0*
_output_shapes
:2+
)stft_1/stft_1_tf.signal.stft/frame/packedк
2stft_1/stft_1_tf.signal.stft/frame/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 24
2stft_1/stft_1_tf.signal.stft/frame/split/split_dim╬
(stft_1/stft_1_tf.signal.stft/frame/splitSplitV1stft_1/stft_1_tf.signal.stft/frame/Shape:output:02stft_1/stft_1_tf.signal.stft/frame/packed:output:0;stft_1/stft_1_tf.signal.stft/frame/split/split_dim:output:0*
T0*

Tlen0*$
_output_shapes
::: *
	num_split2*
(stft_1/stft_1_tf.signal.stft/frame/splitз
0stft_1/stft_1_tf.signal.stft/frame/Reshape/shapeConst*
_output_shapes
: *
dtype0*
valueB 22
0stft_1/stft_1_tf.signal.stft/frame/Reshape/shapeл
2stft_1/stft_1_tf.signal.stft/frame/Reshape/shape_1Const*
_output_shapes
: *
dtype0*
valueB 24
2stft_1/stft_1_tf.signal.stft/frame/Reshape/shape_1Ї
*stft_1/stft_1_tf.signal.stft/frame/ReshapeReshape1stft_1/stft_1_tf.signal.stft/frame/split:output:1;stft_1/stft_1_tf.signal.stft/frame/Reshape/shape_1:output:0*
T0*
_output_shapes
: 2,
*stft_1/stft_1_tf.signal.stft/frame/ReshapeФ
'stft_1/stft_1_tf.signal.stft/frame/SizeConst*
_output_shapes
: *
dtype0*
value	B :2)
'stft_1/stft_1_tf.signal.stft/frame/SizeШ
)stft_1/stft_1_tf.signal.stft/frame/Size_1Const*
_output_shapes
: *
dtype0*
value	B : 2+
)stft_1/stft_1_tf.signal.stft/frame/Size_1х
(stft_1/stft_1_tf.signal.stft/frame/sub_2Sub3stft_1/stft_1_tf.signal.stft/frame/Reshape:output:02stft_1/stft_1_tf.signal.stft/frame_length:output:0*
T0*
_output_shapes
: 2*
(stft_1/stft_1_tf.signal.stft/frame/sub_2ч
+stft_1/stft_1_tf.signal.stft/frame/floordivFloorDiv,stft_1/stft_1_tf.signal.stft/frame/sub_2:z:00stft_1/stft_1_tf.signal.stft/frame_step:output:0*
T0*
_output_shapes
: 2-
+stft_1/stft_1_tf.signal.stft/frame/floordivЦ
(stft_1/stft_1_tf.signal.stft/frame/add/xConst*
_output_shapes
: *
dtype0*
value	B :2*
(stft_1/stft_1_tf.signal.stft/frame/add/x▐
&stft_1/stft_1_tf.signal.stft/frame/addAddV21stft_1/stft_1_tf.signal.stft/frame/add/x:output:0/stft_1/stft_1_tf.signal.stft/frame/floordiv:z:0*
T0*
_output_shapes
: 2(
&stft_1/stft_1_tf.signal.stft/frame/addЮ
,stft_1/stft_1_tf.signal.stft/frame/Maximum/xConst*
_output_shapes
: *
dtype0*
value	B : 2.
,stft_1/stft_1_tf.signal.stft/frame/Maximum/xч
*stft_1/stft_1_tf.signal.stft/frame/MaximumMaximum5stft_1/stft_1_tf.signal.stft/frame/Maximum/x:output:0*stft_1/stft_1_tf.signal.stft/frame/add:z:0*
T0*
_output_shapes
: 2,
*stft_1/stft_1_tf.signal.stft/frame/MaximumЯ
,stft_1/stft_1_tf.signal.stft/frame/gcd/ConstConst*
_output_shapes
: *
dtype0*
value
B :А2.
,stft_1/stft_1_tf.signal.stft/frame/gcd/Constе
/stft_1/stft_1_tf.signal.stft/frame/floordiv_1/yConst*
_output_shapes
: *
dtype0*
value
B :А21
/stft_1/stft_1_tf.signal.stft/frame/floordiv_1/y∙
-stft_1/stft_1_tf.signal.stft/frame/floordiv_1FloorDiv2stft_1/stft_1_tf.signal.stft/frame_length:output:08stft_1/stft_1_tf.signal.stft/frame/floordiv_1/y:output:0*
T0*
_output_shapes
: 2/
-stft_1/stft_1_tf.signal.stft/frame/floordiv_1е
/stft_1/stft_1_tf.signal.stft/frame/floordiv_2/yConst*
_output_shapes
: *
dtype0*
value
B :А21
/stft_1/stft_1_tf.signal.stft/frame/floordiv_2/yў
-stft_1/stft_1_tf.signal.stft/frame/floordiv_2FloorDiv0stft_1/stft_1_tf.signal.stft/frame_step:output:08stft_1/stft_1_tf.signal.stft/frame/floordiv_2/y:output:0*
T0*
_output_shapes
: 2/
-stft_1/stft_1_tf.signal.stft/frame/floordiv_2е
/stft_1/stft_1_tf.signal.stft/frame/floordiv_3/yConst*
_output_shapes
: *
dtype0*
value
B :А21
/stft_1/stft_1_tf.signal.stft/frame/floordiv_3/y·
-stft_1/stft_1_tf.signal.stft/frame/floordiv_3FloorDiv3stft_1/stft_1_tf.signal.stft/frame/Reshape:output:08stft_1/stft_1_tf.signal.stft/frame/floordiv_3/y:output:0*
T0*
_output_shapes
: 2/
-stft_1/stft_1_tf.signal.stft/frame/floordiv_3Ч
(stft_1/stft_1_tf.signal.stft/frame/mul/yConst*
_output_shapes
: *
dtype0*
value
B :А2*
(stft_1/stft_1_tf.signal.stft/frame/mul/y▐
&stft_1/stft_1_tf.signal.stft/frame/mulMul1stft_1/stft_1_tf.signal.stft/frame/floordiv_3:z:01stft_1/stft_1_tf.signal.stft/frame/mul/y:output:0*
T0*
_output_shapes
: 2(
&stft_1/stft_1_tf.signal.stft/frame/mul╩
2stft_1/stft_1_tf.signal.stft/frame/concat/values_1Pack*stft_1/stft_1_tf.signal.stft/frame/mul:z:0*
N*
T0*
_output_shapes
:24
2stft_1/stft_1_tf.signal.stft/frame/concat/values_1в
.stft_1/stft_1_tf.signal.stft/frame/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.stft_1/stft_1_tf.signal.stft/frame/concat/axisь
)stft_1/stft_1_tf.signal.stft/frame/concatConcatV21stft_1/stft_1_tf.signal.stft/frame/split:output:0;stft_1/stft_1_tf.signal.stft/frame/concat/values_1:output:01stft_1/stft_1_tf.signal.stft/frame/split:output:27stft_1/stft_1_tf.signal.stft/frame/concat/axis:output:0*
N*
T0*
_output_shapes
:2+
)stft_1/stft_1_tf.signal.stft/frame/concat│
6stft_1/stft_1_tf.signal.stft/frame/concat_1/values_1/1Const*
_output_shapes
: *
dtype0*
value
B :А28
6stft_1/stft_1_tf.signal.stft/frame/concat_1/values_1/1Ц
4stft_1/stft_1_tf.signal.stft/frame/concat_1/values_1Pack1stft_1/stft_1_tf.signal.stft/frame/floordiv_3:z:0?stft_1/stft_1_tf.signal.stft/frame/concat_1/values_1/1:output:0*
N*
T0*
_output_shapes
:26
4stft_1/stft_1_tf.signal.stft/frame/concat_1/values_1ж
0stft_1/stft_1_tf.signal.stft/frame/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 22
0stft_1/stft_1_tf.signal.stft/frame/concat_1/axisЇ
+stft_1/stft_1_tf.signal.stft/frame/concat_1ConcatV21stft_1/stft_1_tf.signal.stft/frame/split:output:0=stft_1/stft_1_tf.signal.stft/frame/concat_1/values_1:output:01stft_1/stft_1_tf.signal.stft/frame/split:output:29stft_1/stft_1_tf.signal.stft/frame/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2-
+stft_1/stft_1_tf.signal.stft/frame/concat_1и
-stft_1/stft_1_tf.signal.stft/frame/zeros_likeConst*
_output_shapes
:*
dtype0*
valueB: 2/
-stft_1/stft_1_tf.signal.stft/frame/zeros_like▓
2stft_1/stft_1_tf.signal.stft/frame/ones_like/ShapeConst*
_output_shapes
:*
dtype0*
valueB:24
2stft_1/stft_1_tf.signal.stft/frame/ones_like/Shapeк
2stft_1/stft_1_tf.signal.stft/frame/ones_like/ConstConst*
_output_shapes
: *
dtype0*
value	B :24
2stft_1/stft_1_tf.signal.stft/frame/ones_like/ConstГ
,stft_1/stft_1_tf.signal.stft/frame/ones_likeFill;stft_1/stft_1_tf.signal.stft/frame/ones_like/Shape:output:0;stft_1/stft_1_tf.signal.stft/frame/ones_like/Const:output:0*
T0*
_output_shapes
:2.
,stft_1/stft_1_tf.signal.stft/frame/ones_likeА
/stft_1/stft_1_tf.signal.stft/frame/StridedSliceStridedSlicestft_1/transpose:y:06stft_1/stft_1_tf.signal.stft/frame/zeros_like:output:02stft_1/stft_1_tf.signal.stft/frame/concat:output:05stft_1/stft_1_tf.signal.stft/frame/ones_like:output:0*
Index0*
T0*=
_output_shapes+
):'                           21
/stft_1/stft_1_tf.signal.stft/frame/StridedSliceд
,stft_1/stft_1_tf.signal.stft/frame/Reshape_1Reshape8stft_1/stft_1_tf.signal.stft/frame/StridedSlice:output:04stft_1/stft_1_tf.signal.stft/frame/concat_1:output:0*
T0*B
_output_shapes0
.:,                           А2.
,stft_1/stft_1_tf.signal.stft/frame/Reshape_1ж
0stft_1/stft_1_tf.signal.stft/frame/range_1/startConst*
_output_shapes
: *
dtype0*
value	B : 22
0stft_1/stft_1_tf.signal.stft/frame/range_1/startж
0stft_1/stft_1_tf.signal.stft/frame/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :22
0stft_1/stft_1_tf.signal.stft/frame/range_1/deltaм
*stft_1/stft_1_tf.signal.stft/frame/range_1Range9stft_1/stft_1_tf.signal.stft/frame/range_1/start:output:0.stft_1/stft_1_tf.signal.stft/frame/Maximum:z:09stft_1/stft_1_tf.signal.stft/frame/range_1/delta:output:0*#
_output_shapes
:         2,
*stft_1/stft_1_tf.signal.stft/frame/range_1ё
(stft_1/stft_1_tf.signal.stft/frame/mul_1Mul3stft_1/stft_1_tf.signal.stft/frame/range_1:output:01stft_1/stft_1_tf.signal.stft/frame/floordiv_2:z:0*
T0*#
_output_shapes
:         2*
(stft_1/stft_1_tf.signal.stft/frame/mul_1о
4stft_1/stft_1_tf.signal.stft/frame/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :26
4stft_1/stft_1_tf.signal.stft/frame/Reshape_2/shape/1Н
2stft_1/stft_1_tf.signal.stft/frame/Reshape_2/shapePack.stft_1/stft_1_tf.signal.stft/frame/Maximum:z:0=stft_1/stft_1_tf.signal.stft/frame/Reshape_2/shape/1:output:0*
N*
T0*
_output_shapes
:24
2stft_1/stft_1_tf.signal.stft/frame/Reshape_2/shapeД
,stft_1/stft_1_tf.signal.stft/frame/Reshape_2Reshape,stft_1/stft_1_tf.signal.stft/frame/mul_1:z:0;stft_1/stft_1_tf.signal.stft/frame/Reshape_2/shape:output:0*
T0*'
_output_shapes
:         2.
,stft_1/stft_1_tf.signal.stft/frame/Reshape_2ж
0stft_1/stft_1_tf.signal.stft/frame/range_2/startConst*
_output_shapes
: *
dtype0*
value	B : 22
0stft_1/stft_1_tf.signal.stft/frame/range_2/startж
0stft_1/stft_1_tf.signal.stft/frame/range_2/deltaConst*
_output_shapes
: *
dtype0*
value	B :22
0stft_1/stft_1_tf.signal.stft/frame/range_2/deltaж
*stft_1/stft_1_tf.signal.stft/frame/range_2Range9stft_1/stft_1_tf.signal.stft/frame/range_2/start:output:01stft_1/stft_1_tf.signal.stft/frame/floordiv_1:z:09stft_1/stft_1_tf.signal.stft/frame/range_2/delta:output:0*
_output_shapes
:2,
*stft_1/stft_1_tf.signal.stft/frame/range_2о
4stft_1/stft_1_tf.signal.stft/frame/Reshape_3/shape/0Const*
_output_shapes
: *
dtype0*
value	B :26
4stft_1/stft_1_tf.signal.stft/frame/Reshape_3/shape/0Р
2stft_1/stft_1_tf.signal.stft/frame/Reshape_3/shapePack=stft_1/stft_1_tf.signal.stft/frame/Reshape_3/shape/0:output:01stft_1/stft_1_tf.signal.stft/frame/floordiv_1:z:0*
N*
T0*
_output_shapes
:24
2stft_1/stft_1_tf.signal.stft/frame/Reshape_3/shapeВ
,stft_1/stft_1_tf.signal.stft/frame/Reshape_3Reshape3stft_1/stft_1_tf.signal.stft/frame/range_2:output:0;stft_1/stft_1_tf.signal.stft/frame/Reshape_3/shape:output:0*
T0*
_output_shapes

:2.
,stft_1/stft_1_tf.signal.stft/frame/Reshape_3¤
(stft_1/stft_1_tf.signal.stft/frame/add_1AddV25stft_1/stft_1_tf.signal.stft/frame/Reshape_2:output:05stft_1/stft_1_tf.signal.stft/frame/Reshape_3:output:0*
T0*'
_output_shapes
:         2*
(stft_1/stft_1_tf.signal.stft/frame/add_1·
+stft_1/stft_1_tf.signal.stft/frame/GatherV2GatherV25stft_1/stft_1_tf.signal.stft/frame/Reshape_1:output:0,stft_1/stft_1_tf.signal.stft/frame/add_1:z:09stft_1/stft_1_tf.signal.stft/frame/strided_slice:output:0*
Taxis0*
Tindices0*
Tparams0*F
_output_shapes4
2:0                           А2-
+stft_1/stft_1_tf.signal.stft/frame/GatherV2Ж
4stft_1/stft_1_tf.signal.stft/frame/concat_2/values_1Pack.stft_1/stft_1_tf.signal.stft/frame/Maximum:z:02stft_1/stft_1_tf.signal.stft/frame_length:output:0*
N*
T0*
_output_shapes
:26
4stft_1/stft_1_tf.signal.stft/frame/concat_2/values_1ж
0stft_1/stft_1_tf.signal.stft/frame/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 22
0stft_1/stft_1_tf.signal.stft/frame/concat_2/axisЇ
+stft_1/stft_1_tf.signal.stft/frame/concat_2ConcatV21stft_1/stft_1_tf.signal.stft/frame/split:output:0=stft_1/stft_1_tf.signal.stft/frame/concat_2/values_1:output:01stft_1/stft_1_tf.signal.stft/frame/split:output:29stft_1/stft_1_tf.signal.stft/frame/concat_2/axis:output:0*
N*
T0*
_output_shapes
:2-
+stft_1/stft_1_tf.signal.stft/frame/concat_2П
,stft_1/stft_1_tf.signal.stft/frame/Reshape_4Reshape4stft_1/stft_1_tf.signal.stft/frame/GatherV2:output:04stft_1/stft_1_tf.signal.stft/frame/concat_2:output:0*
T0*1
_output_shapes
:         ╖А2.
,stft_1/stft_1_tf.signal.stft/frame/Reshape_4и
1stft_1/stft_1_tf.signal.stft/hann_window/periodicConst*
_output_shapes
: *
dtype0
*
value	B
 Z23
1stft_1/stft_1_tf.signal.stft/hann_window/periodic╥
-stft_1/stft_1_tf.signal.stft/hann_window/CastCast:stft_1/stft_1_tf.signal.stft/hann_window/periodic:output:0*

DstT0*

SrcT0
*
_output_shapes
: 2/
-stft_1/stft_1_tf.signal.stft/hann_window/Castм
3stft_1/stft_1_tf.signal.stft/hann_window/FloorMod/yConst*
_output_shapes
: *
dtype0*
value	B :25
3stft_1/stft_1_tf.signal.stft/hann_window/FloorMod/yЕ
1stft_1/stft_1_tf.signal.stft/hann_window/FloorModFloorMod2stft_1/stft_1_tf.signal.stft/frame_length:output:0<stft_1/stft_1_tf.signal.stft/hann_window/FloorMod/y:output:0*
T0*
_output_shapes
: 23
1stft_1/stft_1_tf.signal.stft/hann_window/FloorModв
.stft_1/stft_1_tf.signal.stft/hann_window/sub/xConst*
_output_shapes
: *
dtype0*
value	B :20
.stft_1/stft_1_tf.signal.stft/hann_window/sub/xЇ
,stft_1/stft_1_tf.signal.stft/hann_window/subSub7stft_1/stft_1_tf.signal.stft/hann_window/sub/x:output:05stft_1/stft_1_tf.signal.stft/hann_window/FloorMod:z:0*
T0*
_output_shapes
: 2.
,stft_1/stft_1_tf.signal.stft/hann_window/subщ
,stft_1/stft_1_tf.signal.stft/hann_window/mulMul1stft_1/stft_1_tf.signal.stft/hann_window/Cast:y:00stft_1/stft_1_tf.signal.stft/hann_window/sub:z:0*
T0*
_output_shapes
: 2.
,stft_1/stft_1_tf.signal.stft/hann_window/mulь
,stft_1/stft_1_tf.signal.stft/hann_window/addAddV22stft_1/stft_1_tf.signal.stft/frame_length:output:00stft_1/stft_1_tf.signal.stft/hann_window/mul:z:0*
T0*
_output_shapes
: 2.
,stft_1/stft_1_tf.signal.stft/hann_window/addж
0stft_1/stft_1_tf.signal.stft/hann_window/sub_1/yConst*
_output_shapes
: *
dtype0*
value	B :22
0stft_1/stft_1_tf.signal.stft/hann_window/sub_1/yї
.stft_1/stft_1_tf.signal.stft/hann_window/sub_1Sub0stft_1/stft_1_tf.signal.stft/hann_window/add:z:09stft_1/stft_1_tf.signal.stft/hann_window/sub_1/y:output:0*
T0*
_output_shapes
: 20
.stft_1/stft_1_tf.signal.stft/hann_window/sub_1╬
/stft_1/stft_1_tf.signal.stft/hann_window/Cast_1Cast2stft_1/stft_1_tf.signal.stft/hann_window/sub_1:z:0*

DstT0*

SrcT0*
_output_shapes
: 21
/stft_1/stft_1_tf.signal.stft/hann_window/Cast_1о
4stft_1/stft_1_tf.signal.stft/hann_window/range/startConst*
_output_shapes
: *
dtype0*
value	B : 26
4stft_1/stft_1_tf.signal.stft/hann_window/range/startо
4stft_1/stft_1_tf.signal.stft/hann_window/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :26
4stft_1/stft_1_tf.signal.stft/hann_window/range/delta╕
.stft_1/stft_1_tf.signal.stft/hann_window/rangeRange=stft_1/stft_1_tf.signal.stft/hann_window/range/start:output:02stft_1/stft_1_tf.signal.stft/frame_length:output:0=stft_1/stft_1_tf.signal.stft/hann_window/range/delta:output:0*
_output_shapes	
:А20
.stft_1/stft_1_tf.signal.stft/hann_window/range╪
/stft_1/stft_1_tf.signal.stft/hann_window/Cast_2Cast7stft_1/stft_1_tf.signal.stft/hann_window/range:output:0*

DstT0*

SrcT0*
_output_shapes	
:А21
/stft_1/stft_1_tf.signal.stft/hann_window/Cast_2е
.stft_1/stft_1_tf.signal.stft/hann_window/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *█╔@20
.stft_1/stft_1_tf.signal.stft/hann_window/Const√
.stft_1/stft_1_tf.signal.stft/hann_window/mul_1Mul7stft_1/stft_1_tf.signal.stft/hann_window/Const:output:03stft_1/stft_1_tf.signal.stft/hann_window/Cast_2:y:0*
T0*
_output_shapes	
:А20
.stft_1/stft_1_tf.signal.stft/hann_window/mul_1■
0stft_1/stft_1_tf.signal.stft/hann_window/truedivRealDiv2stft_1/stft_1_tf.signal.stft/hann_window/mul_1:z:03stft_1/stft_1_tf.signal.stft/hann_window/Cast_1:y:0*
T0*
_output_shapes	
:А22
0stft_1/stft_1_tf.signal.stft/hann_window/truediv┐
,stft_1/stft_1_tf.signal.stft/hann_window/CosCos4stft_1/stft_1_tf.signal.stft/hann_window/truediv:z:0*
T0*
_output_shapes	
:А2.
,stft_1/stft_1_tf.signal.stft/hann_window/Cosй
0stft_1/stft_1_tf.signal.stft/hann_window/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?22
0stft_1/stft_1_tf.signal.stft/hann_window/mul_2/x·
.stft_1/stft_1_tf.signal.stft/hann_window/mul_2Mul9stft_1/stft_1_tf.signal.stft/hann_window/mul_2/x:output:00stft_1/stft_1_tf.signal.stft/hann_window/Cos:y:0*
T0*
_output_shapes	
:А20
.stft_1/stft_1_tf.signal.stft/hann_window/mul_2й
0stft_1/stft_1_tf.signal.stft/hann_window/sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?22
0stft_1/stft_1_tf.signal.stft/hann_window/sub_2/x№
.stft_1/stft_1_tf.signal.stft/hann_window/sub_2Sub9stft_1/stft_1_tf.signal.stft/hann_window/sub_2/x:output:02stft_1/stft_1_tf.signal.stft/hann_window/mul_2:z:0*
T0*
_output_shapes	
:А20
.stft_1/stft_1_tf.signal.stft/hann_window/sub_2Є
 stft_1/stft_1_tf.signal.stft/mulMul5stft_1/stft_1_tf.signal.stft/frame/Reshape_4:output:02stft_1/stft_1_tf.signal.stft/hann_window/sub_2:z:0*
T0*1
_output_shapes
:         ╖А2"
 stft_1/stft_1_tf.signal.stft/mul╝
(stft_1/stft_1_tf.signal.stft/rfft/packedPack0stft_1/stft_1_tf.signal.stft/fft_length:output:0*
N*
T0*
_output_shapes
:2*
(stft_1/stft_1_tf.signal.stft/rfft/packedз
,stft_1/stft_1_tf.signal.stft/rfft/fft_lengthConst*
_output_shapes
:*
dtype0*
valueB:А2.
,stft_1/stft_1_tf.signal.stft/rfft/fft_length▐
!stft_1/stft_1_tf.signal.stft/rfftRFFT$stft_1/stft_1_tf.signal.stft/mul:z:05stft_1/stft_1_tf.signal.stft/rfft/fft_length:output:0*1
_output_shapes
:         ╖Б2#
!stft_1/stft_1_tf.signal.stft/rfftЛ
stft_1/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             2
stft_1/transpose_1/perm┐
stft_1/transpose_1	Transpose*stft_1/stft_1_tf.signal.stft/rfft:output:0 stft_1/transpose_1/perm:output:0*
T0*1
_output_shapes
:         ╖Б2
stft_1/transpose_1{
magnitude_1/Abs
ComplexAbsstft_1/transpose_1:y:0*1
_output_shapes
:         ╖Б2
magnitude_1/Absq
IdentityIdentitymagnitude_1/Abs:y:0*
T0*1
_output_shapes
:         ╖Б2

Identity"
identityIdentity:output:0*,
_input_shapes
:         Ат	:U Q
-
_output_shapes
:         Ат	
 
_user_specified_nameinputs
ч
`
D__inference_reshape_1_layer_call_and_return_conditional_losses_45815

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
valueB	 :Ат	2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/2а
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapeu
ReshapeReshapeinputsReshape/shape:output:0*
T0*-
_output_shapes
:         Ат	2	
Reshapej
IdentityIdentityReshape:output:0*
T0*-
_output_shapes
:         Ат	2

Identity"
identityIdentity:output:0*(
_input_shapes
:         Ат	:Q M
)
_output_shapes
:         Ат	
 
_user_specified_nameinputs
═

▄
C__inference_conv2d_5_layer_call_and_return_conditional_losses_46485

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOpг
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         39 *
paddingSAME*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpИ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         39 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:         39 2
ReluЯ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:         39 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:         39::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         39
 
_user_specified_nameinputs
д
E
)__inference_flatten_1_layer_call_fn_46633

inputs
identity├
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         рv* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_446322
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         рv2

Identity"
identityIdentity:output:0*.
_input_shapes
:          :W S
/
_output_shapes
:          
 
_user_specified_nameinputs
╜
]
F__inference_magnitude_1_layer_call_and_return_conditional_losses_46866
x
identityN
Abs
ComplexAbsx*1
_output_shapes
:         ╖Б2
Abse
IdentityIdentityAbs:y:0*
T0*1
_output_shapes
:         ╖Б2

Identity"
identityIdentity:output:0*0
_input_shapes
:         ╖Б:T P
1
_output_shapes
:         ╖Б

_user_specified_namex
А
c
D__inference_dropout_3_layer_call_and_return_conditional_losses_46712

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *]tС?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:         2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape┤
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:         *
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *П┬ї=2
dropout/GreaterEqual/y╛
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:         2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
л
K
/__inference_max_pooling2d_3_layer_call_fn_43878

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
6:4                                    * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_438722
PartitionedCallП
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
я	
█
B__inference_dense_3_layer_call_and_return_conditional_losses_46644

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	рv@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:         @2
ReluЧ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         @2

Identity"
identityIdentity:output:0*/
_input_shapes
:         рv::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         рv
 
_user_specified_nameinputs
┘

▄
C__inference_conv2d_3_layer_call_and_return_conditional_losses_44334

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpе
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ╖Б*
paddingSAME*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpК
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ╖Б2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:         ╖Б2
Reluб
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:         ╖Б2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:         ╖Б::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:         ╖Б
 
_user_specified_nameinputs
╟
b
D__inference_dropout_3_layer_call_and_return_conditional_losses_46717

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:         2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:         2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
▀
б
,__inference_sequential_1_layer_call_fn_45000
reshape_1_input
unknown
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

unknown_26
identityИвStatefulPartitionedCall█
StatefulPartitionedCallStatefulPartitionedCallreshape_1_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_26*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *6
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_449412
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*Ъ
_input_shapesИ
Е:         Ат	::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
)
_output_shapes
:         Ат	
)
_user_specified_namereshape_1_input
└
є
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_44087

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1▄
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3ь
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
└
є
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_46236

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1▄
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3ь
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
А
c
D__inference_dropout_2_layer_call_and_return_conditional_losses_44679

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ЧЦЦ?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:         @2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape┤
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:         @*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩ>2
dropout/GreaterEqual/y╛
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         @2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         @2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:         @2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:         @2

Identity"
identityIdentity:output:0*&
_input_shapes
:         @:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
╙

▄
C__inference_conv2d_4_layer_call_and_return_conditional_losses_44435

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpд
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         gл*
paddingSAME*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpЙ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         gл2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:         gл2
Reluа
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:         gл2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:         gл::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:         gл
 
_user_specified_nameinputs
╠
Ч
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_46218

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
FusedBatchNormV3н
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue╗
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1Р
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
А
f
J__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_43988

inputs
identityн
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
2	
MaxPoolЗ
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
╝
`
D__inference_flatten_1_layer_call_and_return_conditional_losses_46628

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"    `;  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:         рv2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:         рv2

Identity"
identityIdentity:output:0*.
_input_shapes
:          :W S
/
_output_shapes
:          
 
_user_specified_nameinputs
А
c
D__inference_dropout_2_layer_call_and_return_conditional_losses_46665

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ЧЦЦ?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:         @2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape┤
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:         @*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩ>2
dropout/GreaterEqual/y╛
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         @2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         @2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:         @2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:         @2

Identity"
identityIdentity:output:0*&
_input_shapes
:         @:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
№
є
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_46300

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1╦
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:         gл:::::*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3█
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:         gл2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:         gл::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:         gл
 
_user_specified_nameinputs
╝
J
.__inference_stft_magnitude_layer_call_fn_46050

inputs
identity╤
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ╖Б* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_stft_magnitude_layer_call_and_return_conditional_losses_437592
PartitionedCallv
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:         ╖Б2

Identity"
identityIdentity:output:0*,
_input_shapes
:         Ат	:U Q
-
_output_shapes
:         Ат	
 
_user_specified_nameinputs
■
k
I__inference_stft_magnitude_layer_call_and_return_conditional_losses_43733
stft_1_input
identity▌
stft_1/PartitionedCallPartitionedCallstft_1_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ╖Б* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_stft_1_layer_call_and_return_conditional_losses_437112
stft_1/PartitionedCall 
magnitude_1/PartitionedCallPartitionedCallstft_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ╖Б* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_magnitude_1_layer_call_and_return_conditional_losses_437242
magnitude_1/PartitionedCallВ
IdentityIdentity$magnitude_1/PartitionedCall:output:0*
T0*1
_output_shapes
:         ╖Б2

Identity"
identityIdentity:output:0*,
_input_shapes
:         Ат	:[ W
-
_output_shapes
:         Ат	
&
_user_specified_namestft_1_input
Д
Ч
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_46578

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1╪
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:          : : : : :*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
FusedBatchNormV3н
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue╗
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1■
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:          2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:          ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:          
 
_user_specified_nameinputs"▒L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*╝
serving_defaultи
M
reshape_1_input:
!serving_default_reshape_1_input:0         Ат	;
dense_50
StatefulPartitionedCall:0         tensorflow/serving/predict:│М
╦Й
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
	variables
trainable_variables
regularization_losses
	keras_api

signatures
м__call__
+н&call_and_return_all_conditional_losses
о_default_save_signature"ЖД
_tf_keras_sequentialцГ{"class_name": "Sequential", "name": "sequential_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 160000]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "reshape_1_input"}}, {"class_name": "Reshape", "config": {"name": "reshape_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 160000]}, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [160000, 1]}}}, {"class_name": "Sequential", "config": {"name": "stft_magnitude", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 160000, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "stft_1_input"}}, {"class_name": "STFT", "config": {"name": "stft_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 160000, 1]}, "dtype": "float32", "n_fft": 1024, "win_length": 1024, "hop_length": 512, "window_name": null, "pad_begin": false, "pad_end": false, "input_data_format": "channels_last", "output_data_format": "channels_last"}}, {"class_name": "Magnitude", "config": {"name": "magnitude_1", "trainable": true, "dtype": "float32"}}]}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_4", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_3", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_3", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [3, 3]}, "data_format": "channels_last"}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_5", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_4", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_4", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 3]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 3]}, "data_format": "channels_last"}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_6", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_5", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_5", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 3]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 3]}, "data_format": "channels_last"}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_7", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Flatten", "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_2", "trainable": true, "dtype": "float32", "rate": 0.15, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 16, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_3", "trainable": true, "dtype": "float32", "rate": 0.12, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 160000]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 160000]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "reshape_1_input"}}, {"class_name": "Reshape", "config": {"name": "reshape_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 160000]}, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [160000, 1]}}}, {"class_name": "Sequential", "config": {"name": "stft_magnitude", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 160000, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "stft_1_input"}}, {"class_name": "STFT", "config": {"name": "stft_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 160000, 1]}, "dtype": "float32", "n_fft": 1024, "win_length": 1024, "hop_length": 512, "window_name": null, "pad_begin": false, "pad_end": false, "input_data_format": "channels_last", "output_data_format": "channels_last"}}, {"class_name": "Magnitude", "config": {"name": "magnitude_1", "trainable": true, "dtype": "float32"}}]}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_4", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_3", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_3", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [3, 3]}, "data_format": "channels_last"}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_5", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_4", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_4", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 3]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 3]}, "data_format": "channels_last"}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_6", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_5", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_5", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 3]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 3]}, "data_format": "channels_last"}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_7", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Flatten", "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_2", "trainable": true, "dtype": "float32", "rate": 0.15, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 16, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_3", "trainable": true, "dtype": "float32", "rate": 0.12, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "binary_crossentropy", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "accuracy", "dtype": "float32", "fn": "binary_accuracy"}}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
Ў
	variables
trainable_variables
regularization_losses
	keras_api
п__call__
+░&call_and_return_all_conditional_losses"х
_tf_keras_layer╦{"class_name": "Reshape", "name": "reshape_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 160000]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "reshape_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 160000]}, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [160000, 1]}}}
▐
layer-0
layer-1
	variables
 trainable_variables
!regularization_losses
"	keras_api
▒__call__
+▓&call_and_return_all_conditional_losses"│
_tf_keras_sequentialФ{"class_name": "Sequential", "name": "stft_magnitude", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "stft_magnitude", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 160000, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "stft_1_input"}}, {"class_name": "STFT", "config": {"name": "stft_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 160000, 1]}, "dtype": "float32", "n_fft": 1024, "win_length": 1024, "hop_length": 512, "window_name": null, "pad_begin": false, "pad_end": false, "input_data_format": "channels_last", "output_data_format": "channels_last"}}, {"class_name": "Magnitude", "config": {"name": "magnitude_1", "trainable": true, "dtype": "float32"}}]}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 160000, 1]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "stft_magnitude", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 160000, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "stft_1_input"}}, {"class_name": "STFT", "config": {"name": "stft_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 160000, 1]}, "dtype": "float32", "n_fft": 1024, "win_length": 1024, "hop_length": 512, "window_name": null, "pad_begin": false, "pad_end": false, "input_data_format": "channels_last", "output_data_format": "channels_last"}}, {"class_name": "Magnitude", "config": {"name": "magnitude_1", "trainable": true, "dtype": "float32"}}]}}}
╝	
#axis
	$gamma
%beta
&moving_mean
'moving_variance
(	variables
)trainable_variables
*regularization_losses
+	keras_api
│__call__
+┤&call_and_return_all_conditional_losses"ц
_tf_keras_layer╠{"class_name": "BatchNormalization", "name": "batch_normalization_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_4", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 311, 513, 1]}}
є	

,kernel
-bias
.	variables
/trainable_variables
0regularization_losses
1	keras_api
╡__call__
+╢&call_and_return_all_conditional_losses"╠
_tf_keras_layer▓{"class_name": "Conv2D", "name": "conv2d_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_3", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 311, 513, 1]}}
Б
2	variables
3trainable_variables
4regularization_losses
5	keras_api
╖__call__
+╕&call_and_return_all_conditional_losses"Ё
_tf_keras_layer╓{"class_name": "MaxPooling2D", "name": "max_pooling2d_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_3", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [3, 3]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
╝	
6axis
	7gamma
8beta
9moving_mean
:moving_variance
;	variables
<trainable_variables
=regularization_losses
>	keras_api
╣__call__
+║&call_and_return_all_conditional_losses"ц
_tf_keras_layer╠{"class_name": "BatchNormalization", "name": "batch_normalization_5", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_5", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 8}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 103, 171, 8]}}
Ї	

?kernel
@bias
A	variables
Btrainable_variables
Cregularization_losses
D	keras_api
╗__call__
+╝&call_and_return_all_conditional_losses"═
_tf_keras_layer│{"class_name": "Conv2D", "name": "conv2d_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_4", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 8}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 103, 171, 8]}}
Б
E	variables
Ftrainable_variables
Gregularization_losses
H	keras_api
╜__call__
+╛&call_and_return_all_conditional_losses"Ё
_tf_keras_layer╓{"class_name": "MaxPooling2D", "name": "max_pooling2d_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_4", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 3]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 3]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
╝	
Iaxis
	Jgamma
Kbeta
Lmoving_mean
Mmoving_variance
N	variables
Otrainable_variables
Pregularization_losses
Q	keras_api
┐__call__
+└&call_and_return_all_conditional_losses"ц
_tf_keras_layer╠{"class_name": "BatchNormalization", "name": "batch_normalization_6", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_6", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 51, 57, 16]}}
Ї	

Rkernel
Sbias
T	variables
Utrainable_variables
Vregularization_losses
W	keras_api
┴__call__
+┬&call_and_return_all_conditional_losses"═
_tf_keras_layer│{"class_name": "Conv2D", "name": "conv2d_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_5", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 51, 57, 16]}}
Б
X	variables
Ytrainable_variables
Zregularization_losses
[	keras_api
├__call__
+─&call_and_return_all_conditional_losses"Ё
_tf_keras_layer╓{"class_name": "MaxPooling2D", "name": "max_pooling2d_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_5", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 3]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 3]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
╝	
\axis
	]gamma
^beta
_moving_mean
`moving_variance
a	variables
btrainable_variables
cregularization_losses
d	keras_api
┼__call__
+╞&call_and_return_all_conditional_losses"ц
_tf_keras_layer╠{"class_name": "BatchNormalization", "name": "batch_normalization_7", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_7", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 25, 19, 32]}}
ш
e	variables
ftrainable_variables
gregularization_losses
h	keras_api
╟__call__
+╚&call_and_return_all_conditional_losses"╫
_tf_keras_layer╜{"class_name": "Flatten", "name": "flatten_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
°

ikernel
jbias
k	variables
ltrainable_variables
mregularization_losses
n	keras_api
╔__call__
+╩&call_and_return_all_conditional_losses"╤
_tf_keras_layer╖{"class_name": "Dense", "name": "dense_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 15200}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 15200]}}
ш
o	variables
ptrainable_variables
qregularization_losses
r	keras_api
╦__call__
+╠&call_and_return_all_conditional_losses"╫
_tf_keras_layer╜{"class_name": "Dropout", "name": "dropout_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_2", "trainable": true, "dtype": "float32", "rate": 0.15, "noise_shape": null, "seed": null}}
Є

skernel
tbias
u	variables
vtrainable_variables
wregularization_losses
x	keras_api
═__call__
+╬&call_and_return_all_conditional_losses"╦
_tf_keras_layer▒{"class_name": "Dense", "name": "dense_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 16, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
ш
y	variables
ztrainable_variables
{regularization_losses
|	keras_api
╧__call__
+╨&call_and_return_all_conditional_losses"╫
_tf_keras_layer╜{"class_name": "Dropout", "name": "dropout_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_3", "trainable": true, "dtype": "float32", "rate": 0.12, "noise_shape": null, "seed": null}}
ў

}kernel
~bias
	variables
Аtrainable_variables
Бregularization_losses
В	keras_api
╤__call__
+╥&call_and_return_all_conditional_losses"═
_tf_keras_layer│{"class_name": "Dense", "name": "dense_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16]}}
ш
	Гiter
Дbeta_1
Еbeta_2

Жdecay
Зlearning_rate$mД%mЕ,mЖ-mЗ7mИ8mЙ?mК@mЛJmМKmНRmОSmП]mР^mСimТjmУsmФtmХ}mЦ~mЧ$vШ%vЩ,vЪ-vЫ7vЬ8vЭ?vЮ@vЯJvаKvбRvвSvг]vд^vеivжjvзsvиtvй}vк~vл"
	optimizer
Ў
$0
%1
&2
'3
,4
-5
76
87
98
:9
?10
@11
J12
K13
L14
M15
R16
S17
]18
^19
_20
`21
i22
j23
s24
t25
}26
~27"
trackable_list_wrapper
╢
$0
%1
,2
-3
74
85
?6
@7
J8
K9
R10
S11
]12
^13
i14
j15
s16
t17
}18
~19"
trackable_list_wrapper
 "
trackable_list_wrapper
╙
 Иlayer_regularization_losses
	variables
Йmetrics
Кnon_trainable_variables
trainable_variables
Лlayer_metrics
regularization_losses
Мlayers
м__call__
о_default_save_signature
+н&call_and_return_all_conditional_losses
'н"call_and_return_conditional_losses"
_generic_user_object
-
╙serving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╡
 Нlayer_regularization_losses
Оmetrics
Пnon_trainable_variables
	variables
trainable_variables
Рlayer_metrics
regularization_losses
Сlayers
п__call__
+░&call_and_return_all_conditional_losses
'░"call_and_return_conditional_losses"
_generic_user_object
Є
Т	variables
Уtrainable_variables
Фregularization_losses
Х	keras_api
╘__call__
+╒&call_and_return_all_conditional_losses"▌
_tf_keras_layer├{"class_name": "STFT", "name": "stft_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 160000, 1]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "stft_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 160000, 1]}, "dtype": "float32", "n_fft": 1024, "win_length": 1024, "hop_length": 512, "window_name": null, "pad_begin": false, "pad_end": false, "input_data_format": "channels_last", "output_data_format": "channels_last"}}
┬
Ц	variables
Чtrainable_variables
Шregularization_losses
Щ	keras_api
╓__call__
+╫&call_and_return_all_conditional_losses"н
_tf_keras_layerУ{"class_name": "Magnitude", "name": "magnitude_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "magnitude_1", "trainable": true, "dtype": "float32"}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╡
 Ъlayer_regularization_losses
	variables
Ыmetrics
Ьnon_trainable_variables
 trainable_variables
Эlayer_metrics
!regularization_losses
Юlayers
▒__call__
+▓&call_and_return_all_conditional_losses
'▓"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):'2batch_normalization_4/gamma
(:&2batch_normalization_4/beta
1:/ (2!batch_normalization_4/moving_mean
5:3 (2%batch_normalization_4/moving_variance
<
$0
%1
&2
'3"
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
 "
trackable_list_wrapper
╡
 Яlayer_regularization_losses
аmetrics
бnon_trainable_variables
(	variables
)trainable_variables
вlayer_metrics
*regularization_losses
гlayers
│__call__
+┤&call_and_return_all_conditional_losses
'┤"call_and_return_conditional_losses"
_generic_user_object
):'2conv2d_3/kernel
:2conv2d_3/bias
.
,0
-1"
trackable_list_wrapper
.
,0
-1"
trackable_list_wrapper
 "
trackable_list_wrapper
╡
 дlayer_regularization_losses
еmetrics
жnon_trainable_variables
.	variables
/trainable_variables
зlayer_metrics
0regularization_losses
иlayers
╡__call__
+╢&call_and_return_all_conditional_losses
'╢"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╡
 йlayer_regularization_losses
кmetrics
лnon_trainable_variables
2	variables
3trainable_variables
мlayer_metrics
4regularization_losses
нlayers
╖__call__
+╕&call_and_return_all_conditional_losses
'╕"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):'2batch_normalization_5/gamma
(:&2batch_normalization_5/beta
1:/ (2!batch_normalization_5/moving_mean
5:3 (2%batch_normalization_5/moving_variance
<
70
81
92
:3"
trackable_list_wrapper
.
70
81"
trackable_list_wrapper
 "
trackable_list_wrapper
╡
 оlayer_regularization_losses
пmetrics
░non_trainable_variables
;	variables
<trainable_variables
▒layer_metrics
=regularization_losses
▓layers
╣__call__
+║&call_and_return_all_conditional_losses
'║"call_and_return_conditional_losses"
_generic_user_object
):'2conv2d_4/kernel
:2conv2d_4/bias
.
?0
@1"
trackable_list_wrapper
.
?0
@1"
trackable_list_wrapper
 "
trackable_list_wrapper
╡
 │layer_regularization_losses
┤metrics
╡non_trainable_variables
A	variables
Btrainable_variables
╢layer_metrics
Cregularization_losses
╖layers
╗__call__
+╝&call_and_return_all_conditional_losses
'╝"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╡
 ╕layer_regularization_losses
╣metrics
║non_trainable_variables
E	variables
Ftrainable_variables
╗layer_metrics
Gregularization_losses
╝layers
╜__call__
+╛&call_and_return_all_conditional_losses
'╛"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):'2batch_normalization_6/gamma
(:&2batch_normalization_6/beta
1:/ (2!batch_normalization_6/moving_mean
5:3 (2%batch_normalization_6/moving_variance
<
J0
K1
L2
M3"
trackable_list_wrapper
.
J0
K1"
trackable_list_wrapper
 "
trackable_list_wrapper
╡
 ╜layer_regularization_losses
╛metrics
┐non_trainable_variables
N	variables
Otrainable_variables
└layer_metrics
Pregularization_losses
┴layers
┐__call__
+└&call_and_return_all_conditional_losses
'└"call_and_return_conditional_losses"
_generic_user_object
):' 2conv2d_5/kernel
: 2conv2d_5/bias
.
R0
S1"
trackable_list_wrapper
.
R0
S1"
trackable_list_wrapper
 "
trackable_list_wrapper
╡
 ┬layer_regularization_losses
├metrics
─non_trainable_variables
T	variables
Utrainable_variables
┼layer_metrics
Vregularization_losses
╞layers
┴__call__
+┬&call_and_return_all_conditional_losses
'┬"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╡
 ╟layer_regularization_losses
╚metrics
╔non_trainable_variables
X	variables
Ytrainable_variables
╩layer_metrics
Zregularization_losses
╦layers
├__call__
+─&call_and_return_all_conditional_losses
'─"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):' 2batch_normalization_7/gamma
(:& 2batch_normalization_7/beta
1:/  (2!batch_normalization_7/moving_mean
5:3  (2%batch_normalization_7/moving_variance
<
]0
^1
_2
`3"
trackable_list_wrapper
.
]0
^1"
trackable_list_wrapper
 "
trackable_list_wrapper
╡
 ╠layer_regularization_losses
═metrics
╬non_trainable_variables
a	variables
btrainable_variables
╧layer_metrics
cregularization_losses
╨layers
┼__call__
+╞&call_and_return_all_conditional_losses
'╞"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╡
 ╤layer_regularization_losses
╥metrics
╙non_trainable_variables
e	variables
ftrainable_variables
╘layer_metrics
gregularization_losses
╒layers
╟__call__
+╚&call_and_return_all_conditional_losses
'╚"call_and_return_conditional_losses"
_generic_user_object
!:	рv@2dense_3/kernel
:@2dense_3/bias
.
i0
j1"
trackable_list_wrapper
.
i0
j1"
trackable_list_wrapper
 "
trackable_list_wrapper
╡
 ╓layer_regularization_losses
╫metrics
╪non_trainable_variables
k	variables
ltrainable_variables
┘layer_metrics
mregularization_losses
┌layers
╔__call__
+╩&call_and_return_all_conditional_losses
'╩"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╡
 █layer_regularization_losses
▄metrics
▌non_trainable_variables
o	variables
ptrainable_variables
▐layer_metrics
qregularization_losses
▀layers
╦__call__
+╠&call_and_return_all_conditional_losses
'╠"call_and_return_conditional_losses"
_generic_user_object
 :@2dense_4/kernel
:2dense_4/bias
.
s0
t1"
trackable_list_wrapper
.
s0
t1"
trackable_list_wrapper
 "
trackable_list_wrapper
╡
 рlayer_regularization_losses
сmetrics
тnon_trainable_variables
u	variables
vtrainable_variables
уlayer_metrics
wregularization_losses
фlayers
═__call__
+╬&call_and_return_all_conditional_losses
'╬"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╡
 хlayer_regularization_losses
цmetrics
чnon_trainable_variables
y	variables
ztrainable_variables
шlayer_metrics
{regularization_losses
щlayers
╧__call__
+╨&call_and_return_all_conditional_losses
'╨"call_and_return_conditional_losses"
_generic_user_object
 :2dense_5/kernel
:2dense_5/bias
.
}0
~1"
trackable_list_wrapper
.
}0
~1"
trackable_list_wrapper
 "
trackable_list_wrapper
╖
 ъlayer_regularization_losses
ыmetrics
ьnon_trainable_variables
	variables
Аtrainable_variables
эlayer_metrics
Бregularization_losses
юlayers
╤__call__
+╥&call_and_return_all_conditional_losses
'╥"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_list_wrapper
0
я0
Ё1"
trackable_list_wrapper
X
&0
'1
92
:3
L4
M5
_6
`7"
trackable_list_wrapper
 "
trackable_dict_wrapper
ж
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
╕
 ёlayer_regularization_losses
Єmetrics
єnon_trainable_variables
Т	variables
Уtrainable_variables
Їlayer_metrics
Фregularization_losses
їlayers
╘__call__
+╒&call_and_return_all_conditional_losses
'╒"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
 Ўlayer_regularization_losses
ўmetrics
°non_trainable_variables
Ц	variables
Чtrainable_variables
∙layer_metrics
Шregularization_losses
·layers
╓__call__
+╫&call_and_return_all_conditional_losses
'╫"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
&0
'1"
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
90
:1"
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
L0
M1"
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
_0
`1"
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
 "
trackable_list_wrapper
┐

√total

№count
¤	variables
■	keras_api"Д
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
 

 total

Аcount
Б
_fn_kwargs
В	variables
Г	keras_api"│
_tf_keras_metricШ{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "binary_accuracy"}}
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
:  (2total
:  (2count
0
√0
№1"
trackable_list_wrapper
.
¤	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
 0
А1"
trackable_list_wrapper
.
В	variables"
_generic_user_object
.:,2"Adam/batch_normalization_4/gamma/m
-:+2!Adam/batch_normalization_4/beta/m
.:,2Adam/conv2d_3/kernel/m
 :2Adam/conv2d_3/bias/m
.:,2"Adam/batch_normalization_5/gamma/m
-:+2!Adam/batch_normalization_5/beta/m
.:,2Adam/conv2d_4/kernel/m
 :2Adam/conv2d_4/bias/m
.:,2"Adam/batch_normalization_6/gamma/m
-:+2!Adam/batch_normalization_6/beta/m
.:, 2Adam/conv2d_5/kernel/m
 : 2Adam/conv2d_5/bias/m
.:, 2"Adam/batch_normalization_7/gamma/m
-:+ 2!Adam/batch_normalization_7/beta/m
&:$	рv@2Adam/dense_3/kernel/m
:@2Adam/dense_3/bias/m
%:#@2Adam/dense_4/kernel/m
:2Adam/dense_4/bias/m
%:#2Adam/dense_5/kernel/m
:2Adam/dense_5/bias/m
.:,2"Adam/batch_normalization_4/gamma/v
-:+2!Adam/batch_normalization_4/beta/v
.:,2Adam/conv2d_3/kernel/v
 :2Adam/conv2d_3/bias/v
.:,2"Adam/batch_normalization_5/gamma/v
-:+2!Adam/batch_normalization_5/beta/v
.:,2Adam/conv2d_4/kernel/v
 :2Adam/conv2d_4/bias/v
.:,2"Adam/batch_normalization_6/gamma/v
-:+2!Adam/batch_normalization_6/beta/v
.:, 2Adam/conv2d_5/kernel/v
 : 2Adam/conv2d_5/bias/v
.:, 2"Adam/batch_normalization_7/gamma/v
-:+ 2!Adam/batch_normalization_7/beta/v
&:$	рv@2Adam/dense_3/kernel/v
:@2Adam/dense_3/bias/v
%:#@2Adam/dense_4/kernel/v
:2Adam/dense_4/bias/v
%:#2Adam/dense_5/kernel/v
:2Adam/dense_5/bias/v
■2√
,__inference_sequential_1_layer_call_fn_45000
,__inference_sequential_1_layer_call_fn_45139
,__inference_sequential_1_layer_call_fn_45802
,__inference_sequential_1_layer_call_fn_45741└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
ъ2ч
G__inference_sequential_1_layer_call_and_return_conditional_losses_45456
G__inference_sequential_1_layer_call_and_return_conditional_losses_45680
G__inference_sequential_1_layer_call_and_return_conditional_losses_44782
G__inference_sequential_1_layer_call_and_return_conditional_losses_44860└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
ш2х
 __inference__wrapped_model_43598└
Л▓З
FullArgSpec
argsЪ 
varargsjargs
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *0в-
+К(
reshape_1_input         Ат	
╙2╨
)__inference_reshape_1_layer_call_fn_45820в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ю2ы
D__inference_reshape_1_layer_call_and_return_conditional_losses_45815в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
Ж2Г
.__inference_stft_magnitude_layer_call_fn_43751
.__inference_stft_magnitude_layer_call_fn_46050
.__inference_stft_magnitude_layer_call_fn_43762
.__inference_stft_magnitude_layer_call_fn_46045└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
Є2я
I__inference_stft_magnitude_layer_call_and_return_conditional_losses_43739
I__inference_stft_magnitude_layer_call_and_return_conditional_losses_43733
I__inference_stft_magnitude_layer_call_and_return_conditional_losses_45930
I__inference_stft_magnitude_layer_call_and_return_conditional_losses_46040└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
Ц2У
5__inference_batch_normalization_4_layer_call_fn_46114
5__inference_batch_normalization_4_layer_call_fn_46165
5__inference_batch_normalization_4_layer_call_fn_46101
5__inference_batch_normalization_4_layer_call_fn_46178┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
В2 
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_46134
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_46070
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_46152
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_46088┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
╥2╧
(__inference_conv2d_3_layer_call_fn_46198в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
э2ъ
C__inference_conv2d_3_layer_call_and_return_conditional_losses_46189в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
Ч2Ф
/__inference_max_pooling2d_3_layer_call_fn_43878р
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *@в=
;К84                                    
▓2п
J__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_43872р
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *@в=
;К84                                    
Ц2У
5__inference_batch_normalization_5_layer_call_fn_46326
5__inference_batch_normalization_5_layer_call_fn_46313
5__inference_batch_normalization_5_layer_call_fn_46249
5__inference_batch_normalization_5_layer_call_fn_46262┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
В2 
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_46300
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_46236
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_46218
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_46282┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
╥2╧
(__inference_conv2d_4_layer_call_fn_46346в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
э2ъ
C__inference_conv2d_4_layer_call_and_return_conditional_losses_46337в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
Ч2Ф
/__inference_max_pooling2d_4_layer_call_fn_43994р
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *@в=
;К84                                    
▓2п
J__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_43988р
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *@в=
;К84                                    
Ц2У
5__inference_batch_normalization_6_layer_call_fn_46410
5__inference_batch_normalization_6_layer_call_fn_46474
5__inference_batch_normalization_6_layer_call_fn_46461
5__inference_batch_normalization_6_layer_call_fn_46397┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
В2 
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_46366
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_46430
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_46384
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_46448┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
╥2╧
(__inference_conv2d_5_layer_call_fn_46494в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
э2ъ
C__inference_conv2d_5_layer_call_and_return_conditional_losses_46485в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
Ч2Ф
/__inference_max_pooling2d_5_layer_call_fn_44110р
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *@в=
;К84                                    
▓2п
J__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_44104р
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *@в=
;К84                                    
Ц2У
5__inference_batch_normalization_7_layer_call_fn_46545
5__inference_batch_normalization_7_layer_call_fn_46622
5__inference_batch_normalization_7_layer_call_fn_46609
5__inference_batch_normalization_7_layer_call_fn_46558┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
В2 
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_46596
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_46532
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_46578
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_46514┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
╙2╨
)__inference_flatten_1_layer_call_fn_46633в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ю2ы
D__inference_flatten_1_layer_call_and_return_conditional_losses_46628в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╤2╬
'__inference_dense_3_layer_call_fn_46653в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ь2щ
B__inference_dense_3_layer_call_and_return_conditional_losses_46644в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
Р2Н
)__inference_dropout_2_layer_call_fn_46675
)__inference_dropout_2_layer_call_fn_46680┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
╞2├
D__inference_dropout_2_layer_call_and_return_conditional_losses_46670
D__inference_dropout_2_layer_call_and_return_conditional_losses_46665┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
╤2╬
'__inference_dense_4_layer_call_fn_46700в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ь2щ
B__inference_dense_4_layer_call_and_return_conditional_losses_46691в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
Р2Н
)__inference_dropout_3_layer_call_fn_46727
)__inference_dropout_3_layer_call_fn_46722┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
╞2├
D__inference_dropout_3_layer_call_and_return_conditional_losses_46712
D__inference_dropout_3_layer_call_and_return_conditional_losses_46717┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
╤2╬
'__inference_dense_5_layer_call_fn_46747в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ь2щ
B__inference_dense_5_layer_call_and_return_conditional_losses_46738в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╥B╧
#__inference_signature_wrapper_45210reshape_1_input"Ф
Н▓Й
FullArgSpec
argsЪ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╦2╚
&__inference_stft_1_layer_call_fn_46861Э
Ф▓Р
FullArgSpec
argsЪ
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ц2у
A__inference_stft_1_layer_call_and_return_conditional_losses_46856Э
Ф▓Р
FullArgSpec
argsЪ
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╨2═
+__inference_magnitude_1_layer_call_fn_46871Э
Ф▓Р
FullArgSpec
argsЪ
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ы2ш
F__inference_magnitude_1_layer_call_and_return_conditional_losses_46866Э
Ф▓Р
FullArgSpec
argsЪ
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 ▓
 __inference__wrapped_model_43598Н$%&',-789:?@JKLMRS]^_`ijst}~:в7
0в-
+К(
reshape_1_input         Ат	
к "1к.
,
dense_5!К
dense_5         ы
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_46070Ц$%&'MвJ
Cв@
:К7
inputs+                           
p
к "?в<
5К2
0+                           
Ъ ы
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_46088Ц$%&'MвJ
Cв@
:К7
inputs+                           
p 
к "?в<
5К2
0+                           
Ъ ╩
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_46134v$%&'=в:
3в0
*К'
inputs         ╖Б
p
к "/в,
%К"
0         ╖Б
Ъ ╩
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_46152v$%&'=в:
3в0
*К'
inputs         ╖Б
p 
к "/в,
%К"
0         ╖Б
Ъ ├
5__inference_batch_normalization_4_layer_call_fn_46101Й$%&'MвJ
Cв@
:К7
inputs+                           
p
к "2К/+                           ├
5__inference_batch_normalization_4_layer_call_fn_46114Й$%&'MвJ
Cв@
:К7
inputs+                           
p 
к "2К/+                           в
5__inference_batch_normalization_4_layer_call_fn_46165i$%&'=в:
3в0
*К'
inputs         ╖Б
p
к ""К         ╖Бв
5__inference_batch_normalization_4_layer_call_fn_46178i$%&'=в:
3в0
*К'
inputs         ╖Б
p 
к ""К         ╖Бы
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_46218Ц789:MвJ
Cв@
:К7
inputs+                           
p
к "?в<
5К2
0+                           
Ъ ы
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_46236Ц789:MвJ
Cв@
:К7
inputs+                           
p 
к "?в<
5К2
0+                           
Ъ ╚
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_46282t789:<в9
2в/
)К&
inputs         gл
p
к ".в+
$К!
0         gл
Ъ ╚
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_46300t789:<в9
2в/
)К&
inputs         gл
p 
к ".в+
$К!
0         gл
Ъ ├
5__inference_batch_normalization_5_layer_call_fn_46249Й789:MвJ
Cв@
:К7
inputs+                           
p
к "2К/+                           ├
5__inference_batch_normalization_5_layer_call_fn_46262Й789:MвJ
Cв@
:К7
inputs+                           
p 
к "2К/+                           а
5__inference_batch_normalization_5_layer_call_fn_46313g789:<в9
2в/
)К&
inputs         gл
p
к "!К         gла
5__inference_batch_normalization_5_layer_call_fn_46326g789:<в9
2в/
)К&
inputs         gл
p 
к "!К         gлы
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_46366ЦJKLMMвJ
Cв@
:К7
inputs+                           
p
к "?в<
5К2
0+                           
Ъ ы
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_46384ЦJKLMMвJ
Cв@
:К7
inputs+                           
p 
к "?в<
5К2
0+                           
Ъ ╞
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_46430rJKLM;в8
1в.
(К%
inputs         39
p
к "-в*
#К 
0         39
Ъ ╞
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_46448rJKLM;в8
1в.
(К%
inputs         39
p 
к "-в*
#К 
0         39
Ъ ├
5__inference_batch_normalization_6_layer_call_fn_46397ЙJKLMMвJ
Cв@
:К7
inputs+                           
p
к "2К/+                           ├
5__inference_batch_normalization_6_layer_call_fn_46410ЙJKLMMвJ
Cв@
:К7
inputs+                           
p 
к "2К/+                           Ю
5__inference_batch_normalization_6_layer_call_fn_46461eJKLM;в8
1в.
(К%
inputs         39
p
к " К         39Ю
5__inference_batch_normalization_6_layer_call_fn_46474eJKLM;в8
1в.
(К%
inputs         39
p 
к " К         39ы
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_46514Ц]^_`MвJ
Cв@
:К7
inputs+                            
p
к "?в<
5К2
0+                            
Ъ ы
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_46532Ц]^_`MвJ
Cв@
:К7
inputs+                            
p 
к "?в<
5К2
0+                            
Ъ ╞
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_46578r]^_`;в8
1в.
(К%
inputs          
p
к "-в*
#К 
0          
Ъ ╞
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_46596r]^_`;в8
1в.
(К%
inputs          
p 
к "-в*
#К 
0          
Ъ ├
5__inference_batch_normalization_7_layer_call_fn_46545Й]^_`MвJ
Cв@
:К7
inputs+                            
p
к "2К/+                            ├
5__inference_batch_normalization_7_layer_call_fn_46558Й]^_`MвJ
Cв@
:К7
inputs+                            
p 
к "2К/+                            Ю
5__inference_batch_normalization_7_layer_call_fn_46609e]^_`;в8
1в.
(К%
inputs          
p
к " К          Ю
5__inference_batch_normalization_7_layer_call_fn_46622e]^_`;в8
1в.
(К%
inputs          
p 
к " К          ╖
C__inference_conv2d_3_layer_call_and_return_conditional_losses_46189p,-9в6
/в,
*К'
inputs         ╖Б
к "/в,
%К"
0         ╖Б
Ъ П
(__inference_conv2d_3_layer_call_fn_46198c,-9в6
/в,
*К'
inputs         ╖Б
к ""К         ╖Б╡
C__inference_conv2d_4_layer_call_and_return_conditional_losses_46337n?@8в5
.в+
)К&
inputs         gл
к ".в+
$К!
0         gл
Ъ Н
(__inference_conv2d_4_layer_call_fn_46346a?@8в5
.в+
)К&
inputs         gл
к "!К         gл│
C__inference_conv2d_5_layer_call_and_return_conditional_losses_46485lRS7в4
-в*
(К%
inputs         39
к "-в*
#К 
0         39 
Ъ Л
(__inference_conv2d_5_layer_call_fn_46494_RS7в4
-в*
(К%
inputs         39
к " К         39 г
B__inference_dense_3_layer_call_and_return_conditional_losses_46644]ij0в-
&в#
!К
inputs         рv
к "%в"
К
0         @
Ъ {
'__inference_dense_3_layer_call_fn_46653Pij0в-
&в#
!К
inputs         рv
к "К         @в
B__inference_dense_4_layer_call_and_return_conditional_losses_46691\st/в,
%в"
 К
inputs         @
к "%в"
К
0         
Ъ z
'__inference_dense_4_layer_call_fn_46700Ost/в,
%в"
 К
inputs         @
к "К         в
B__inference_dense_5_layer_call_and_return_conditional_losses_46738\}~/в,
%в"
 К
inputs         
к "%в"
К
0         
Ъ z
'__inference_dense_5_layer_call_fn_46747O}~/в,
%в"
 К
inputs         
к "К         д
D__inference_dropout_2_layer_call_and_return_conditional_losses_46665\3в0
)в&
 К
inputs         @
p
к "%в"
К
0         @
Ъ д
D__inference_dropout_2_layer_call_and_return_conditional_losses_46670\3в0
)в&
 К
inputs         @
p 
к "%в"
К
0         @
Ъ |
)__inference_dropout_2_layer_call_fn_46675O3в0
)в&
 К
inputs         @
p
к "К         @|
)__inference_dropout_2_layer_call_fn_46680O3в0
)в&
 К
inputs         @
p 
к "К         @д
D__inference_dropout_3_layer_call_and_return_conditional_losses_46712\3в0
)в&
 К
inputs         
p
к "%в"
К
0         
Ъ д
D__inference_dropout_3_layer_call_and_return_conditional_losses_46717\3в0
)в&
 К
inputs         
p 
к "%в"
К
0         
Ъ |
)__inference_dropout_3_layer_call_fn_46722O3в0
)в&
 К
inputs         
p
к "К         |
)__inference_dropout_3_layer_call_fn_46727O3в0
)в&
 К
inputs         
p 
к "К         й
D__inference_flatten_1_layer_call_and_return_conditional_losses_46628a7в4
-в*
(К%
inputs          
к "&в#
К
0         рv
Ъ Б
)__inference_flatten_1_layer_call_fn_46633T7в4
-в*
(К%
inputs          
к "К         рv▒
F__inference_magnitude_1_layer_call_and_return_conditional_losses_46866g4в1
*в'
%К"
x         ╖Б
к "/в,
%К"
0         ╖Б
Ъ Й
+__inference_magnitude_1_layer_call_fn_46871Z4в1
*в'
%К"
x         ╖Б
к ""К         ╖Бэ
J__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_43872ЮRвO
HвE
CК@
inputs4                                    
к "HвE
>К;
04                                    
Ъ ┼
/__inference_max_pooling2d_3_layer_call_fn_43878СRвO
HвE
CК@
inputs4                                    
к ";К84                                    э
J__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_43988ЮRвO
HвE
CК@
inputs4                                    
к "HвE
>К;
04                                    
Ъ ┼
/__inference_max_pooling2d_4_layer_call_fn_43994СRвO
HвE
CК@
inputs4                                    
к ";К84                                    э
J__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_44104ЮRвO
HвE
CК@
inputs4                                    
к "HвE
>К;
04                                    
Ъ ┼
/__inference_max_pooling2d_5_layer_call_fn_44110СRвO
HвE
CК@
inputs4                                    
к ";К84                                    и
D__inference_reshape_1_layer_call_and_return_conditional_losses_45815`1в.
'в$
"К
inputs         Ат	
к "+в(
!К
0         Ат	
Ъ А
)__inference_reshape_1_layer_call_fn_45820S1в.
'в$
"К
inputs         Ат	
к "К         Ат	╒
G__inference_sequential_1_layer_call_and_return_conditional_losses_44782Й$%&',-789:?@JKLMRS]^_`ijst}~Bв?
8в5
+К(
reshape_1_input         Ат	
p

 
к "%в"
К
0         
Ъ ╒
G__inference_sequential_1_layer_call_and_return_conditional_losses_44860Й$%&',-789:?@JKLMRS]^_`ijst}~Bв?
8в5
+К(
reshape_1_input         Ат	
p 

 
к "%в"
К
0         
Ъ ╠
G__inference_sequential_1_layer_call_and_return_conditional_losses_45456А$%&',-789:?@JKLMRS]^_`ijst}~9в6
/в,
"К
inputs         Ат	
p

 
к "%в"
К
0         
Ъ ╠
G__inference_sequential_1_layer_call_and_return_conditional_losses_45680А$%&',-789:?@JKLMRS]^_`ijst}~9в6
/в,
"К
inputs         Ат	
p 

 
к "%в"
К
0         
Ъ м
,__inference_sequential_1_layer_call_fn_45000|$%&',-789:?@JKLMRS]^_`ijst}~Bв?
8в5
+К(
reshape_1_input         Ат	
p

 
к "К         м
,__inference_sequential_1_layer_call_fn_45139|$%&',-789:?@JKLMRS]^_`ijst}~Bв?
8в5
+К(
reshape_1_input         Ат	
p 

 
к "К         г
,__inference_sequential_1_layer_call_fn_45741s$%&',-789:?@JKLMRS]^_`ijst}~9в6
/в,
"К
inputs         Ат	
p

 
к "К         г
,__inference_sequential_1_layer_call_fn_45802s$%&',-789:?@JKLMRS]^_`ijst}~9в6
/в,
"К
inputs         Ат	
p 

 
к "К         ╚
#__inference_signature_wrapper_45210а$%&',-789:?@JKLMRS]^_`ijst}~MвJ
в 
Cк@
>
reshape_1_input+К(
reshape_1_input         Ат	"1к.
,
dense_5!К
dense_5         и
A__inference_stft_1_layer_call_and_return_conditional_losses_46856c0в-
&в#
!К
x         Ат	
к "/в,
%К"
0         ╖Б
Ъ А
&__inference_stft_1_layer_call_fn_46861V0в-
&в#
!К
x         Ат	
к ""К         ╖Б├
I__inference_stft_magnitude_layer_call_and_return_conditional_losses_43733vCв@
9в6
,К)
stft_1_input         Ат	
p

 
к "/в,
%К"
0         ╖Б
Ъ ├
I__inference_stft_magnitude_layer_call_and_return_conditional_losses_43739vCв@
9в6
,К)
stft_1_input         Ат	
p 

 
к "/в,
%К"
0         ╖Б
Ъ ╜
I__inference_stft_magnitude_layer_call_and_return_conditional_losses_45930p=в:
3в0
&К#
inputs         Ат	
p

 
к "/в,
%К"
0         ╖Б
Ъ ╜
I__inference_stft_magnitude_layer_call_and_return_conditional_losses_46040p=в:
3в0
&К#
inputs         Ат	
p 

 
к "/в,
%К"
0         ╖Б
Ъ Ы
.__inference_stft_magnitude_layer_call_fn_43751iCв@
9в6
,К)
stft_1_input         Ат	
p

 
к ""К         ╖БЫ
.__inference_stft_magnitude_layer_call_fn_43762iCв@
9в6
,К)
stft_1_input         Ат	
p 

 
к ""К         ╖БХ
.__inference_stft_magnitude_layer_call_fn_46045c=в:
3в0
&К#
inputs         Ат	
p

 
к ""К         ╖БХ
.__inference_stft_magnitude_layer_call_fn_46050c=в:
3в0
&К#
inputs         Ат	
p 

 
к ""К         ╖Б