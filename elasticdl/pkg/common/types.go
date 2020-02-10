package common

import (
	"elasticdl.org/elasticdl/pkg/proto"
	"reflect"
)

// const alias for dtype
const (
	Invalid = proto.ElementType_DT_INVALID
	Int8    = proto.ElementType_DT_INT8
	Int16   = proto.ElementType_DT_INT16
	Int32   = proto.ElementType_DT_INT32
	Int64   = proto.ElementType_DT_INT64
	Float16 = proto.ElementType_DT_FLOAT16
	Float32 = proto.ElementType_DT_FLOAT32
	Float64 = proto.ElementType_DT_FLOAT64
	Bool    = proto.ElementType_DT_BOOL
)

// DtypeSize Dtype -> size
var DtypeSize = make(map[proto.ElementType]int32)

// DtypeToSliceType Dtype -> reflect.Type
var DtypeToSliceType = make(map[proto.ElementType]reflect.Type)

// SliceTypeToDtype reflect.Type -> Dtype
var SliceTypeToDtype = make(map[reflect.Type]proto.ElementType)

func init() {
	DtypeSize[proto.ElementType_DT_INVALID] = 1
	DtypeSize[proto.ElementType_DT_INT8] = 1
	DtypeSize[proto.ElementType_DT_INT16] = 2
	DtypeSize[proto.ElementType_DT_INT32] = 4
	DtypeSize[proto.ElementType_DT_INT64] = 8
	DtypeSize[proto.ElementType_DT_FLOAT16] = 2
	DtypeSize[proto.ElementType_DT_FLOAT32] = 4
	DtypeSize[proto.ElementType_DT_FLOAT64] = 8
	DtypeSize[proto.ElementType_DT_BOOL] = 1

	DtypeToSliceType[proto.ElementType_DT_INVALID] = reflect.TypeOf([]byte{0})
	DtypeToSliceType[proto.ElementType_DT_INT8] = reflect.TypeOf([]int8{0})
	DtypeToSliceType[proto.ElementType_DT_INT16] = reflect.TypeOf([]int16{0})
	DtypeToSliceType[proto.ElementType_DT_INT32] = reflect.TypeOf([]int32{0})
	DtypeToSliceType[proto.ElementType_DT_INT64] = reflect.TypeOf([]int64{0})
	//DtypeToSliceType[proto.ElementType_DT_FLOAT16] = reflect.TypeOf([]float16{0})
	DtypeToSliceType[proto.ElementType_DT_FLOAT32] = reflect.TypeOf([]float32{0})
	DtypeToSliceType[proto.ElementType_DT_FLOAT64] = reflect.TypeOf([]float64{0})
	DtypeToSliceType[proto.ElementType_DT_BOOL] = reflect.TypeOf([]bool{true})

	SliceTypeToDtype[reflect.TypeOf([]byte{0})] = proto.ElementType_DT_INVALID
	SliceTypeToDtype[reflect.TypeOf([]int8{0})] = proto.ElementType_DT_INT8
	SliceTypeToDtype[reflect.TypeOf([]int16{0})] = proto.ElementType_DT_INT16
	SliceTypeToDtype[reflect.TypeOf([]int32{0})] = proto.ElementType_DT_INT32
	SliceTypeToDtype[reflect.TypeOf([]int64{0})] = proto.ElementType_DT_INT64
	//SliceTypeToDtype[reflect.TypeOf([]float16{0})] = proto.ElementType_DT_FLOAT16
	SliceTypeToDtype[reflect.TypeOf([]float32{0})] = proto.ElementType_DT_FLOAT32
	SliceTypeToDtype[reflect.TypeOf([]float64{0})] = proto.ElementType_DT_FLOAT64
	SliceTypeToDtype[reflect.TypeOf([]bool{true})] = proto.ElementType_DT_BOOL
}
