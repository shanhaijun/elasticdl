package ps

import (
	"elasticdl.org/elasticdl/pkg/common"
	"elasticdl.org/elasticdl/pkg/proto"
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestPsModelInit(t *testing.T) {
	d1 := []int64{2, 3}
	v1 := []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}
	t1 := common.NewTensor(v1, d1)

	d2 := []int64{2, 2}
	v2 := []float32{1.0, 2.0, 1.1, 2.2}
	t2 := common.NewTensor(v2, d2)

	p := NewModel()
	p.DenseParameters["t1"] = t1
	p.DenseParameters["t2"] = t2

	assert.Len(t, p.DenseParameters, 2)
	assert.Contains(t, p.DenseParameters, "t1")
	assert.Contains(t, p.DenseParameters, "t2")

	assert.Equal(t, p.GetDenseParameter("t1").Dims, d1)
	assert.Equal(t, p.GetDenseParameter("t2").Dims, d2)
	assert.Nil(t, p.GetDenseParameter("t3"))
}

func TestParameterInitFromModelPB(t *testing.T) {
	var modelPB = proto.Model{
		Version:             int32(1),
		IndexedSlices:       make(map[string]*proto.IndexedSlices),
		EmbeddingTableInfos: []*proto.EmbeddingTableInfo{},
	}

	i1 := []int64{1, 3, 5}
	d1 := []int64{3, 2}
	v1 := []float64{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}
	t1 := common.NewTensor(v1, d1)
	is1 := common.NewIndexedSlices(t1, i1)
	modelPB.IndexedSlices["e1"] = is1

	var epb = proto.EmbeddingTableInfo{
		Name:        "e1",
		Dim:         2,
		Initializer: "zero",
		Dtype:       common.Float64,
	}

	modelPB.EmbeddingTableInfos = append(modelPB.EmbeddingTableInfos, &epb)

	model := NewModel()
	err := model.InitFromModelPB(&modelPB)

	assert.Nil(t, err)
	assert.Contains(t, model.EmbeddingTables, "e1")

	e1 := model.GetEmbeddingTable("e1")
	assert.Equal(t, int64(2), e1.Dim)
	assert.Equal(t, 3, len(e1.EmbeddingVectors))

	ev1 := e1.GetEmbeddingVector(1)
	assert.True(t, common.CompareFloatArray([]float64{1.0, 2.0}, common.Slice(ev1).([]float64), 0.0001))

	ev3 := e1.GetEmbeddingVector(3)
	assert.True(t, common.CompareFloatArray([]float64{3.0, 4.0}, common.Slice(ev3).([]float64), 0.0001))

	ev5 := e1.GetEmbeddingVector(5)
	assert.True(t, common.CompareFloatArray([]float64{5.0, 6.0}, common.Slice(ev5).([]float64), 0.0001))

	is2 := e1.GetEmbeddingVectors([]int64{1, 3, 3, 4})
	assert.True(t, common.CompareFloatArray([]float64{1.0, 2.0, 3.0, 4.0, 3.0, 4.0, 0, 0}, common.Slice(is2).([]float64), 0.0001))
}
