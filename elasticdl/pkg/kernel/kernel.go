package kernel

// #cgo LDFLAGS: -L./capi -lkernel_api -lm
// #include "capi/kernel_api.h"
import "C"
import (
	"elasticdl.org/elasticdl/pkg/common"
	"elasticdl.org/elasticdl/pkg/proto"
	"fmt"
	"unsafe"
)

// SGD kernel
func SGD(grad *proto.Tensor, param *proto.Tensor, lr float32) error {
	if len(grad.Content) != len(param.Content) {
		return fmt.Errorf("grad Value size not equal to param")
	}
	gradPtr := (*C.float)(unsafe.Pointer(&grad.Content[0]))
	paramPtr := (*C.float)(unsafe.Pointer(&param.Content[0]))
	length := len(grad.Content) / int(common.DtypeSize[grad.Dtype])
	C.SGD(gradPtr, paramPtr, C.float(lr), C.longlong(length))
	return nil
}

// SparseSGD kernel
func SparseSGD(grad *proto.IndexedSlices, param *common.EmbeddingTable, lr float32) error {
	if grad.ConcatedVectors.Dims[1] != param.Dim {
		return fmt.Errorf("grad width is not equal to embedding dim")
	}
	MergeIndexedSlices(grad)
	for i, index := range grad.Ids {
		if index == -1 {
			continue
		}
		vector := param.GetEmbeddingVector(index)
		subGrad := common.RowOfTensor(grad.ConcatedVectors, int64(i))
		SGD(subGrad, vector, lr)
	}
	return nil
}

// Adam kernel
func Adam(grad *proto.Tensor, param *proto.Tensor, m *proto.Tensor, v *proto.Tensor,
	lr float32, step int64, beta1 float32, beta2 float32,
	epsilon float32, amsgrad bool, maxSquare *proto.Tensor) {
	gradPtr := (*C.float)(unsafe.Pointer(&grad.Content[0]))
	paramPtr := (*C.float)(unsafe.Pointer(&param.Content[0]))
	mPtr := (*C.float)(unsafe.Pointer(&m.Content[0]))
	vPtr := (*C.float)(unsafe.Pointer(&v.Content[0]))
	length := len(grad.Content) / int(common.DtypeSize[grad.Dtype])
	if amsgrad {
		maxSquarePtr := (*C.float)(unsafe.Pointer(&maxSquare.Content[0]))
		C.Adam(gradPtr, paramPtr, mPtr, vPtr, C.float(lr), C.longlong(length),
			C.longlong(step), C.float(beta1), C.float(beta2), C.float(epsilon),
			maxSquarePtr)
	} else {
		C.Adam(gradPtr, paramPtr, mPtr, vPtr, C.float(lr), C.longlong(length),
			C.longlong(step), C.float(beta1), C.float(beta2), C.float(epsilon), nil)
	}
}

// Sum kernel
func Sum(tensors []*proto.Tensor) {
	if len(tensors) < 2 {
		return
	}
	tosum := []uintptr{}
	for _, t := range tensors {
		tosum = append(tosum, (uintptr)(unsafe.Pointer(&t.Content[0])))
	}
	length := len(tensors[0].Content) / int(common.DtypeSize[tensors[0].Dtype])
	C.Sum((*unsafe.Pointer)(unsafe.Pointer(&tosum[0])), C.longlong(len(tensors)), C.longlong(length))
}

func mapIdsToRows(indices []int64) map[int64][]int64 {
	result := make(map[int64][]int64)
	for i, item := range indices {
		value, ok := result[item]
		if !ok {
			result[item] = []int64{int64(i)}
		} else {
			result[item] = append(value, int64(i))
		}
	}
	return result
}

// MergeIndexedSlices Merge duplicate indexed gradients
func MergeIndexedSlices(grad *proto.IndexedSlices) {
	ids2rows := mapIdsToRows(grad.Ids)
	indices := grad.Ids
	for _, rows := range ids2rows {
		if len(rows) < 2 {
			continue
		}
		tosum := []*proto.Tensor{common.RowOfTensor(grad.ConcatedVectors, rows[0])}
		for _, row := range rows[1:] {
			indices[row] = -1
			tosum = append(tosum, common.RowOfTensor(grad.ConcatedVectors, row))
		}
		Sum(tosum)
	}
}

// SparseAdam kernel
func SparseAdam(grad *proto.IndexedSlices, param *common.EmbeddingTable, m *common.EmbeddingTable, v *common.EmbeddingTable, lr float32, step int64, beta1 float32, beta2 float32, epsilon float32, amsgrad bool, maxSquare *common.EmbeddingTable) error {
	if grad.ConcatedVectors.Dims[1] != param.Dim {
		return fmt.Errorf("grad width is not equal to embedding dim")
	}
	MergeIndexedSlices(grad)
	for i, index := range grad.Ids {
		if index == -1 {
			continue
		}
		subgrad := common.RowOfTensor(grad.ConcatedVectors, int64(i))
		subparam := param.GetEmbeddingVector(index)
		subm := m.GetEmbeddingVector(index)
		subv := v.GetEmbeddingVector(index)
		var submaxs *proto.Tensor = nil
		if amsgrad {
			submaxs = maxSquare.GetEmbeddingVector(index)
		}
		Adam(subgrad, subparam, subm, subv, lr, step, beta1, beta2, epsilon, amsgrad, submaxs)
	}
	return nil
}
