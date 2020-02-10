package common

import (
	"elasticdl.org/elasticdl/pkg/proto"
)

// EmbeddingTable struct
type EmbeddingTable struct {
	Dim             int64
	Initializer     string
	EmbeddingVector map[int64]*proto.Tensor
	Dtype           proto.ElementType
}

// NewIndexedSlices return proto.IndexedSlices
func NewIndexedSlices(t *proto.Tensor, indices []int64) *proto.IndexedSlices {
	var i = proto.IndexedSlices{
		ConcatedVectors: t,
		Ids:             indices,
	}
	return &i
}

// NewEmbeddingTable creates an embedding table instance
func NewEmbeddingTable(dim int64, initializer string, dtype proto.ElementType) *EmbeddingTable {
	var e = EmbeddingTable{
		Dim:             dim,
		Initializer:     initializer,
		EmbeddingVector: make(map[int64]*proto.Tensor),
		Dtype:           dtype,
	}
	return &e
}

// GetEmbeddingVector returns embedding vector giving an index
func (e *EmbeddingTable) GetEmbeddingVector(index int64) *proto.Tensor {
	if value, ok := e.EmbeddingVector[index]; ok {
		return value
	}
	newVector := NewEmptyVector(e.Dim, e.Dtype)
	e.EmbeddingVector[index] = newVector
	return newVector
}

// GetEmbeddingVectors returns embedding vectors giving an array of indices
func (e *EmbeddingTable) GetEmbeddingVectors(indices []int64) *proto.IndexedSlices {
	dim := []int64{int64(len(indices)), e.Dim}
	tensor := NewEmptyTensor(dim, e.Dtype)
	for i, index := range indices {
		SetTensorRow(tensor, int64(i), e.GetEmbeddingVector(index))
	}
	var i = proto.IndexedSlices{
		ConcatedVectors: tensor,
		Ids:             indices,
	}
	return &i
}

// SetEmbeddingVectors sets (indices, value) pair to embedding vector
func (e *EmbeddingTable) SetEmbeddingVectors(is *proto.IndexedSlices) error {
	for i, index := range is.Ids {
		value := e.GetEmbeddingVector(index)
		copy(value.Content, RowOfTensor(is.ConcatedVectors, int64(i)).Content)
	}
	return nil
}
