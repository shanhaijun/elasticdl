package common

import (
	"elasticdl.org/elasticdl/pkg/proto"
	"reflect"
	"unsafe"
)

// NewEmptyTensor create an empty n-dim tensor
func NewEmptyTensor(dim []int64, dtype proto.ElementType) *proto.Tensor {
	var t = proto.Tensor{
		Content: make([]byte, DimProduct(dim)*int64(DtypeSize[dtype])),
		Dims:    dim,
		Dtype:   dtype,
	}
	return &t
}

// NewTensor create a n-dim tensor using exsiting slice
func NewTensor(slice interface{}, dim []int64) *proto.Tensor {
	v := reflect.ValueOf(slice)
	length := v.Len()
	dtype := SliceTypeToDtype[reflect.TypeOf(slice)]
	bytelen := length * int(DtypeSize[dtype])
	if int64(length) != DimProduct(dim) {
		return nil
	}
	sliceHeader := reflect.SliceHeader{
		Data: uintptr(unsafe.Pointer(v.Pointer())),
		Cap:  int(bytelen),
		Len:  int(bytelen),
	}
	var t = proto.Tensor{
		Content: *(*[]byte)(unsafe.Pointer(&sliceHeader)),
		Dims:    dim,
		Dtype:   dtype,
	}
	return &t
}

// NewEmptyVector create an empty 1-dim tensor
func NewEmptyVector(dim int64, dtype proto.ElementType) *proto.Tensor {
	var t = proto.Tensor{
		Content: make([]byte, dim*int64(DtypeSize[dtype])),
		Dims:    []int64{dim},
		Dtype:   dtype,
	}
	return &t
}

// NewVector create an empty 1-dim tensor
func NewVector(slice interface{}) *proto.Tensor {
	v := reflect.ValueOf(slice)
	length := v.Len()
	dtype := SliceTypeToDtype[reflect.TypeOf(slice)]
	bytelen := length * int(DtypeSize[dtype])
	if v.Len() != length {
		return nil
	}
	sliceHeader := reflect.SliceHeader{
		Data: uintptr(unsafe.Pointer(v.Pointer())),
		Cap:  int(bytelen),
		Len:  int(bytelen),
	}
	var t = proto.Tensor{
		Content: *(*[]byte)(unsafe.Pointer(&sliceHeader)),
		Dims:    []int64{int64(length)},
		Dtype:   dtype,
	}
	return &t
}

// DimProduct get the number of the elements of a tensor of this dim
func DimProduct(dim []int64) int64 {
	var size int64 = 1
	for _, d := range dim {
		size *= d
	}
	return size
}

// SubTensor get the part reference of the tensor
func SubTensor(t *proto.Tensor, begin int64, length int64) *proto.Tensor {
	dsize := int64(DtypeSize[t.Dtype])
	begin *= dsize
	var subt = proto.Tensor{
		Content: t.Content[begin : begin+length*dsize],
		Dims:    []int64{length},
		Dtype:   t.Dtype,
	}
	return &subt
}

// RowOfTensor get the row reference of a 2-dim tensor
func RowOfTensor(t *proto.Tensor, idx int64) *proto.Tensor {
	if len(t.Dims) != 2 || idx >= t.Dims[0] {
		return nil
	}
	begin := t.Dims[1] * idx
	return SubTensor(t, begin, t.Dims[1])
}

// SetSubTensor set a vector to an index of tensor
func SetSubTensor(t *proto.Tensor, begin int64, length int64, val *proto.Tensor) {
	dsize := int64(DtypeSize[t.Dtype])
	begin *= dsize
	length *= dsize
	copy(t.Content[begin:begin+length], val.Content)
}

// SetTensorRow set a vector to an index of tensor
func SetTensorRow(t *proto.Tensor, idx int64, vec *proto.Tensor) {
	if len(t.Dims) != 2 || idx >= t.Dims[0] {
		return
	}
	begin := t.Dims[1] * idx
	SetSubTensor(t, begin, t.Dims[1], vec)
}

// Slice gives a Slice interface to the Tensor data
func Slice(t *proto.Tensor) interface{} {
	length := int(DimProduct(t.Dims))
	sliceHeader := reflect.SliceHeader{
		Data: uintptr(unsafe.Pointer(&t.Content[0])),
		Cap:  length,
		Len:  length,
	}
	val := reflect.NewAt(DtypeToSliceType[t.Dtype], unsafe.Pointer(&sliceHeader)).Elem()
	return val.Interface()
}
