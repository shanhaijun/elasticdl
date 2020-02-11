package ps

import (
	"elasticdl.org/elasticdl/pkg/common"
	"elasticdl.org/elasticdl/pkg/proto"
)

// Model definition
type Model struct {
	Version         int32
	InitStatus      bool
	DenseParameters map[string]*proto.Tensor
	EmbeddingTables map[string]*common.EmbeddingTable
}

// NewModel creates a parameter instance
func NewModel() *Model {
	var model = Model{
		DenseParameters: make(map[string]*proto.Tensor),
		EmbeddingTables: make(map[string]*common.EmbeddingTable),
	}
	return &model
}

// GetDenseParameter returns non-embedding tensor pointer
func (model *Model) GetDenseParameter(name string) *proto.Tensor {
	if value, ok := model.DenseParameters[name]; ok {
		return value
	}
	return nil
}

// GetEmbeddingTable returns embedding table pointer
func (model *Model) GetEmbeddingTable(name string) *common.EmbeddingTable {
	if value, ok := model.EmbeddingTables[name]; ok {
		return value
	}
	return nil
}

// SetEmbeddingTableInfo sets embedding table info of an embedding param
func (model *Model) SetEmbeddingTableInfo(info *proto.EmbeddingTableInfo) *common.EmbeddingTable {
	if _, ok := model.EmbeddingTables[info.Name]; ok {
		return nil
	}
	t := common.NewEmbeddingTable(info.Dim, info.Initializer, info.Dtype)
	model.EmbeddingTables[info.Name] = t
	return t
}

// InitFromModelPB inits a PsModel instance from model PB to Parameter
func (model *Model) InitFromModelPB(pb *proto.Model) error {
	for _, v := range pb.EmbeddingTableInfos {
		model.SetEmbeddingTableInfo(v)
	}
	for name, v := range pb.DenseParameters {
		model.DenseParameters[name] = v
	}
	for name, v := range pb.IndexedSlices {
		model.EmbeddingTables[name].SetEmbeddingVectors(v)
	}
	if pb.Version >= 0 {
		model.Version = pb.Version
	}
	return nil
}
