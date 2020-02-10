package ps

import "elasticdl.org/elasticdl/pkg/proto"
import "elasticdl.org/elasticdl/pkg/common"

// PsModel definition
type PsModel struct {
	Version         int32
	InitStatus      bool
	DenseParameters map[string]*proto.Tensor
	EmbeddingTables map[string]*common.EmbeddingTable
	Dtype           proto.ElementDtype
}

// NewPsModel creates a parameter instance
func NewPsModel(dtype proto.ElementDtype) *PsModel {
	var p = PsModel{
		DenseParameters: make(map[string]*proto.Tensor),
		EmbeddingTables: make(map[string]*common.EmbeddingTable),
		Dtype          : dtype
	}
	return &p
}

// GetDenseParameter returns non-embedding tensor pointer
func (p *PsModel) GetDenseParameter(name string) *proto.Tensor {
	if value, ok := p.DenseParameters[name]; ok {
		return value
	}
	return nil
}

// GetEmbeddingParam returns embedding table pointer
func (p *PsModel) GetEmbeddingTable(name string) *common.EmbeddingTable {
	if value, ok := p.EmbeddingTables[name]; ok {
		return value
	}
	return nil
}

// SetEmbeddingParamInfo sets embedding table info of an embedding param
func (p *PsModel) SetEmbeddingParamInfo(info proto.EmbeddingTableInfo) *common.EmbeddingTable {
	if _, ok := p.EmbeddingParam[info.name]; ok {
		return nil
	}
	t := common.NewEmbeddingTable(info.Dim, info.initializer, p.Dtype)
	p.EmbeddingParam[info.name] = t
	return t
}

// InitFromModelPB inits a PsModel instance from model PB to Parameter
func (p *PsModel) InitFromModelPB(pb *proto.Model) error {
	for _, v := range pb.EmbeddingTableInfos {
		p.SetEmbeddingParamInfo(v)
	}
	for name, v := range pb.DenseParameters {
		p.NonEmbeddingParam[name] = v
	}
	if pb.Version >= 0 {
		p.Version = pb.Version
	}
	return nil
}

