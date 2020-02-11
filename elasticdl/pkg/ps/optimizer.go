package ps

import "elasticdl.org/elasticdl/pkg/common"
import "elasticdl.org/elasticdl/pkg/kernel"
import "elasticdl.org/elasticdl/pkg/proto"
import "fmt"

// Optimizer interface
type Optimizer interface {
	GetLR() float32
	ApplyDenseGradients(map[string]*proto.Tensor, *Model) error
	ApplySparseGradients(map[string]*proto.IndexedSlices, *Model) error
	ApplyGradients(*proto.Model, *Model) error
	InitFromModelPB(*proto.Model) error
}

// BaseOptimizer struct
type BaseOptimizer struct {
	lr   float32
	step int64
}

// SGDOptimizer struct
type SGDOptimizer struct {
	BaseOptimizer
}

// GetLR returns learning rate SGD
func (opt *SGDOptimizer) GetLR() float32 {
	return opt.lr
}

// ApplyDenseGradients applies gradients to parameters
func (opt *SGDOptimizer) ApplyDenseGradients(grads map[string]*proto.Tensor, p *Model) error {
	for name, grad := range grads {
		denseParam := p.GetDenseParameter(name)
		if denseParam != nil {
			kernel.SGD(grad, denseParam, opt.GetLR())
		} else {
			return fmt.Errorf("grad %s not in Parameter", name)
		}
	}
	return nil
}

// ApplySparseGradients applies gradients to parameters
func (opt *SGDOptimizer) ApplySparseGradients(grads map[string]*proto.IndexedSlices, p *Model) error {
	for name, grad := range grads {
		sparseParam := p.GetEmbeddingTable(name)
		if sparseParam != nil {
			kernel.SparseSGD(grad, sparseParam, opt.GetLR())
		} else {
			return fmt.Errorf("grad %s not in Parameter", name)
		}
	}
	return nil
}

// ApplyGradients applies gradients to parameters
func (opt *SGDOptimizer) ApplyGradients(model *proto.Model, p *Model) error {
	opt.step++
	err := opt.ApplyDenseGradients(model.DenseParameters, p)
	if err != nil {
		return err
	}
	err = opt.ApplySparseGradients(model.IndexedSlices, p)
	if err != nil {
		return err
	}
	return nil
}

// InitFromModelPB Nothing to Init
func (opt *SGDOptimizer) InitFromModelPB(pb *proto.Model) error {
	return nil
}

// NewSGDOptimizer creates a SGD optimizer instance
func NewSGDOptimizer(lr float32) *SGDOptimizer {
	var opt = SGDOptimizer{
		BaseOptimizer: BaseOptimizer{
			lr:   lr,
			step: 0,
		},
	}
	return &opt
}

// AdamOptimizer struct
type AdamOptimizer struct {
	BaseOptimizer
	beta1     float32
	beta2     float32
	epsilon   float32
	amsgrad   bool
	m         *Model
	v         *Model
	maxSquare *Model
}

// ApplyDenseGradients applies gradients to parameters
func (opt *AdamOptimizer) ApplyDenseGradients(grads map[string]*proto.Tensor, p *Model) error {
	for name, grad := range grads {
		denseParam := p.GetDenseParameter(name)
		if denseParam != nil {
			m := opt.m.GetDenseParameter(name)
			v := opt.v.GetDenseParameter(name)
			if opt.amsgrad {
				ms := opt.maxSquare.GetDenseParameter(name)
				kernel.Adam(grad, denseParam, m, v, opt.lr, opt.step,
					opt.beta1, opt.beta2, opt.epsilon, true, ms)
			} else {
				kernel.Adam(grad, denseParam, m, v, opt.lr, opt.step,
					opt.beta1, opt.beta2, opt.epsilon, false, nil)
			}
		} else {
			return fmt.Errorf("grad %s not in Parameter", name)
		}
	}
	return nil
}

// ApplySparseGradients applies gradients to parameters
func (opt *AdamOptimizer) ApplySparseGradients(grads map[string]*proto.IndexedSlices, p *Model) error {
	for name, grad := range grads {
		sparseParam := p.GetEmbeddingTable(name)
		if sparseParam != nil {
			m := opt.m.GetEmbeddingTable(name)
			v := opt.v.GetEmbeddingTable(name)
			if opt.amsgrad {
				ms := opt.maxSquare.GetEmbeddingTable(name)
				kernel.SparseAdam(grad, sparseParam, m, v, opt.lr, opt.step,
					opt.beta1, opt.beta2, opt.epsilon, true, ms)
			} else {
				kernel.SparseAdam(grad, sparseParam, m, v, opt.lr, opt.step,
					opt.beta1, opt.beta2, opt.epsilon, false, nil)
			}
		} else {
			return fmt.Errorf("grad %s not in Parameter", name)
		}
	}
	return nil
}

// ApplyGradients applies gradients to parameters
func (opt *AdamOptimizer) ApplyGradients(model *proto.Model, p *Model) error {
	opt.step++
	err := opt.ApplyDenseGradients(model.DenseParameters, p)
	if err != nil {
		return err
	}
	err = opt.ApplySparseGradients(model.IndexedSlices, p)
	if err != nil {
		return err
	}
	return nil
}

// NewAdamOptimizer creates a Adam optimizer instance
func NewAdamOptimizer(lr float32, beta1 float32, beta2 float32, epsilon float32, amsgrad bool) *AdamOptimizer {
	var opt = AdamOptimizer{
		BaseOptimizer: BaseOptimizer{
			lr:   lr,
			step: 0,
		},
		beta1:     beta1,
		beta2:     beta2,
		epsilon:   epsilon,
		amsgrad:   amsgrad,
		m:         NewModel(),
		v:         NewModel(),
		maxSquare: NewModel(),
	}
	return &opt
}

// InitEmbeddingTables set m,v,maxSquare embedding of AdamOptimizer (TODO)
func (opt *AdamOptimizer) InitEmbeddingTables(infos []*proto.EmbeddingTableInfo) {
	for _, info := range infos {
		opt.m.SetEmbeddingTableInfo(info)
		opt.v.SetEmbeddingTableInfo(info)
		opt.maxSquare.SetEmbeddingTableInfo(info)
	}
}

// InitDenseParameters set m,v,maxSquare non-embedding of AdamOptimizer
func (opt *AdamOptimizer) InitDenseParameters(tensors map[string]*proto.Tensor) {
	for name, tensor := range tensors {
		dims := tensor.Dims
		dtype := tensor.Dtype
		opt.m.DenseParameters[name] = common.NewEmptyTensor(dims, dtype)
		opt.v.DenseParameters[name] = common.NewEmptyTensor(dims, dtype)
		opt.maxSquare.DenseParameters[name] = common.NewEmptyTensor(dims, dtype)
	}
}

// InitFromModelPB set m,v,maxSquare non-embedding of AdamOptimizer
func (opt *AdamOptimizer) InitFromModelPB(pb *proto.Model) error {
	opt.InitDenseParameters(pb.DenseParameters)
	opt.InitEmbeddingTables(pb.EmbeddingTableInfos)
	return nil
}

// NewOptimizer creates an optimizer instance
func NewOptimizer(opt string, lr float32) Optimizer {
	// TODO(qijun) only support SGD now
	switch opt {
	case "SGD":
		return NewSGDOptimizer(lr)
	case "Adam":
		return nil
	default:
		return nil
	}
}
