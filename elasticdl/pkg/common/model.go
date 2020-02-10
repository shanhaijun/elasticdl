package common

import "elasticdl.org/elasticdl/pkg/proto"

// PsModel definition
type PsModel struct {
	Version         int32
	InitStatus      bool
	DenseParameters map[string]*proto.Tensor
	EmbeddingTables map[string]*EmbeddingTable
}

// NewPsModel creates a parameter instance
func NewPsModel() *PsModel {
	var p = PsModel{
		DenseParameters: make(map[string]*proto.Tensor),
		EmbeddingTables: make(map[string]*EmbeddingTable),
	}
	return &p
}
