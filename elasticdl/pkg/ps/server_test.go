package ps

import (
	"context"
	"elasticdl.org/elasticdl/pkg/common"
	"elasticdl.org/elasticdl/pkg/proto"
	"github.com/stretchr/testify/assert"
	"google.golang.org/grpc"
	"log"
	"math/rand"
	"testing"
	"time"
)

const (
	ADDR string = "localhost:12345"
)

func createClient() (proto.PserverClient, context.Context, *grpc.ClientConn, context.CancelFunc) {
	conn, err := grpc.Dial(ADDR, grpc.WithInsecure(), grpc.WithBlock())
	if err != nil {
		log.Fatalf("did not connect: %v", err)
	}
	c := proto.NewPserverClient(conn)
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	return c, ctx, conn, cancel
}

func TestPushDenseParameters(t *testing.T) {
	// Create a PS server
	serverDone := make(chan bool)
	s := NewServer(0, "SGD", 0.1)
	gs := s.Run(ADDR, serverDone)
	client, ctx, conn, cancel := createClient()
	defer conn.Close()
	defer cancel()

	var request = &proto.Model{
		DenseParameters: make(map[string]*proto.Tensor),
		IndexedSlices:   make(map[string]*proto.IndexedSlices),
	}
	// dense embedding param
	a := make([]float32, 10)
	b := make([]float32, 10)
	for i := 0; i < 10; i++ {
		a[i] = rand.Float32()
		b[i] = rand.Float32()
	}
	d := []int64{2, 5}
	t1 := common.NewTensor(a, d) // t1
	t2 := common.NewTensor(b, d) // t2

	request.DenseParameters["t1"] = t1
	request.DenseParameters["t2"] = t2

	_, err := client.PushModel(ctx, request)

	if err != nil {
		t.Errorf("Failed to push model")
	}

	assert.True(t, s.Model.InitStatus)
	assert.Len(t, s.Model.DenseParameters, 2)
	assert.Contains(t, s.Model.DenseParameters, "t1")
	assert.Contains(t, s.Model.DenseParameters, "t2")
	assert.True(t, common.CompareFloatArray(a, common.Slice(s.Model.GetDenseParameter("t1")).([]float32), 0.0001))
	assert.True(t, common.CompareFloatArray(b, common.Slice(s.Model.GetDenseParameter("t2")).([]float32), 0.0001))
	gs.Stop()
}

func TestPushEmbeddingInfo(t *testing.T) {
	// Create a PS server
	serverDone := make(chan bool)
	s := NewServer(0, "SGD", 0.1)
	gs := s.Run(ADDR, serverDone)
	client, ctx, conn, cancel := createClient()
	defer conn.Close()
	defer cancel()

	var request = &proto.Model{
		DenseParameters:     make(map[string]*proto.Tensor),
		IndexedSlices:       make(map[string]*proto.IndexedSlices),
		EmbeddingTableInfos: []*proto.EmbeddingTableInfo{},
	}
	// embedding table info
	var epb = &proto.EmbeddingTableInfo{
		Name:        "e1",
		Dim:         2,
		Initializer: "zero",
		Dtype:       common.Float32,
	}
	request.EmbeddingTableInfos = append(request.EmbeddingTableInfos, epb)

	_, err := client.PushModel(ctx, request)
	if err != nil {
		t.Errorf("Failed to push embedding vector info")
	}

	assert.Contains(t, s.Model.EmbeddingTables, "e1")
	assert.Equal(t, int64(2), s.Model.GetEmbeddingTable("e1").Dim)
	gs.Stop()
}

func TestPushGradient(t *testing.T) {
	// Create a PS server
	serverDone := make(chan bool)
	s := NewServer(0, "SGD", 0.1)
	gs := s.Run(ADDR, serverDone)
	client, ctx, conn, cancel := createClient()
	defer conn.Close()
	defer cancel()

	var request = &proto.Model{
		DenseParameters:     make(map[string]*proto.Tensor),
		IndexedSlices:       make(map[string]*proto.IndexedSlices),
		EmbeddingTableInfos: []*proto.EmbeddingTableInfo{},
	}

	// embedding table info
	var epb = &proto.EmbeddingTableInfo{
		Name:        "e1",
		Dim:         10,
		Initializer: "zero",
		Dtype:       common.Float32,
	}
	request.EmbeddingTableInfos = append(request.EmbeddingTableInfos, epb)

	// dense embedding param
	a := make([]float32, 10)
	b := make([]float32, 10)
	c := make([]float32, 10)
	for i := 0; i < 10; i++ {
		a[i] = rand.Float32()
		b[i] = rand.Float32()
		c[i] = rand.Float32()
	}
	d := []int64{2, 5}
	t1 := common.NewTensor(a, d) // t1
	t2 := common.NewTensor(b, d) // t2

	ed := []int64{1, 10}
	e1 := common.NewIndexedSlices(common.NewTensor(c, ed), []int64{1})

	request.DenseParameters["t1"] = t1
	request.DenseParameters["t2"] = t2
	request.IndexedSlices["e1"] = e1

	client.PushModel(ctx, request)

	_, err := client.PushGradients(ctx, request)
	if err != nil {
		t.Errorf("Failed to push embedding vector info")
	}

	expectedt1 := make([]float32, 10, 10)
	expectedt2 := make([]float32, 10, 10)
	expectede1 := make([]float32, 10, 10)

	for i := 0; i < 10; i++ {
		expectedt1[i] = a[i] - 0.1*a[i]
		expectedt2[i] = b[i] - 0.1*b[i]
		expectede1[i] = c[i] - 0.1*c[i]
	}

	assert.True(t, common.CompareFloatArray(expectedt1, common.Slice(s.Model.GetDenseParameter("t1")).([]float32), 0.0001))
	assert.True(t, common.CompareFloatArray(expectedt2, common.Slice(s.Model.GetDenseParameter("t2")).([]float32), 0.0001))
	assert.True(t, common.CompareFloatArray(expectede1, common.Slice(s.Model.GetEmbeddingTable("e1").GetEmbeddingVector(1)).([]float32), 0.0001))
	gs.Stop()
}

func TestPullEmbeddingTables(t *testing.T) {
	// Create a PS server
	serverDone := make(chan bool)
	s := NewServer(0, "SGD", 0.1)
	gs := s.Run(ADDR, serverDone)
	client, ctx, conn, cancel := createClient()
	defer conn.Close()
	defer cancel()

	var request = &proto.Model{
		DenseParameters:     make(map[string]*proto.Tensor),
		IndexedSlices:       make(map[string]*proto.IndexedSlices),
		EmbeddingTableInfos: []*proto.EmbeddingTableInfo{},
	}

	// embedding table info
	var epb = &proto.EmbeddingTableInfo{
		Name:        "e1",
		Dim:         10,
		Initializer: "zero",
		Dtype:       common.Float32,
	}
	request.EmbeddingTableInfos = append(request.EmbeddingTableInfos, epb)

	// dense embedding param
	a := make([]float32, 10)
	b := make([]float32, 10)
	c := make([]float32, 10)
	for i := 0; i < 10; i++ {
		a[i] = rand.Float32()
		b[i] = rand.Float32()
		c[i] = rand.Float32()
	}
	d := []int64{2, 5}
	t1 := common.NewTensor(a, d) // t1
	t2 := common.NewTensor(b, d) // t2

	ed := []int64{1, 10}
	e1 := common.NewIndexedSlices(common.NewTensor(c, ed), []int64{1})

	request.DenseParameters["t1"] = t1
	request.DenseParameters["t2"] = t2
	request.IndexedSlices["e1"] = e1

	client.PushModel(ctx, request)

	pr := &proto.PullEmbeddingTableRequest{
		Name: "e1",
		Ids:  []int64{1},
	}

	resp, _ := client.PullEmbeddingTable(ctx, pr)
	assert.True(t, common.CompareFloatArray(c, common.Slice(resp).([]float32), 0.0001))
	gs.Stop()
}

func TestPullDenseParameters(t *testing.T) {
	// Create a PS server
	serverDone := make(chan bool)
	s := NewServer(0, "SGD", 0.1)
	gs := s.Run(ADDR, serverDone)
	client, ctx, conn, cancel := createClient()
	defer conn.Close()
	defer cancel()

	var request = &proto.Model{
		DenseParameters:     make(map[string]*proto.Tensor),
		IndexedSlices:       make(map[string]*proto.IndexedSlices),
		EmbeddingTableInfos: []*proto.EmbeddingTableInfo{},
	}

	// embedding table info
	var epb = &proto.EmbeddingTableInfo{
		Name:        "e1",
		Dim:         10,
		Initializer: "zero",
		Dtype:       common.Float32,
	}
	request.EmbeddingTableInfos = append(request.EmbeddingTableInfos, epb)

	// dense embedding param
	a := make([]float32, 10)
	b := make([]float32, 10)
	c := make([]float32, 10)
	for i := 0; i < 10; i++ {
		a[i] = rand.Float32()
		b[i] = rand.Float32()
		c[i] = rand.Float32()
	}
	d := []int64{2, 5}
	t1 := common.NewTensor(a, d) // t1
	t2 := common.NewTensor(b, d) // t2

	ed := []int64{1, 10}
	e1 := common.NewIndexedSlices(common.NewTensor(c, ed), []int64{1})

	request.DenseParameters["t1"] = t1
	request.DenseParameters["t2"] = t2
	request.IndexedSlices["e1"] = e1

	client.PushModel(ctx, request)

	pr := &proto.PullDenseParametersRequest{
		Version: 0,
	}

	resp, _ := client.PullDenseParameters(ctx, pr)
	assert.Equal(t, true, resp.Initialized)
	assert.Equal(t, int32(0), resp.Version)
	assert.True(t, common.CompareFloatArray(a, common.Slice(resp.DenseParameters["t1"]).([]float32), 0.0001))
	assert.True(t, common.CompareFloatArray(b, common.Slice(resp.DenseParameters["t2"]).([]float32), 0.0001))
	gs.Stop()
}
