package ps

import (
	"context"
	//"elasticdl.org/elasticdl/pkg/common"
	"elasticdl.org/elasticdl/pkg/proto"
	//"fmt"
	"github.com/golang/protobuf/ptypes/empty"
	"google.golang.org/grpc"
	"log"
	"net"
	"sync"
)

// Server defines servicer of ps
type Server struct {
	Model       *Model
	Optimizer   Optimizer
	ID          int // a zero-based successive integer number
	lock        sync.Mutex
	versionLock sync.Mutex
}

// NewServer creates a Server instance
func NewServer(ID int, opt string, lr float32) *Server {
	return &Server{
		Model:     NewModel(),
		Optimizer: NewOptimizer(opt, lr),
		ID:        ID}
}

// PullDenseParameters pulls dense parameter from server
func (s *Server) PullDenseParameters(ctx context.Context, in *proto.PullDenseParametersRequest) (*proto.PullDenseParametersResponse, error) {
	if !s.Model.InitStatus {
		return &proto.PullDenseParametersResponse{Initialized: false}, nil
	}
	var resp = proto.PullDenseParametersResponse{
		Initialized:     true,
		Version:         s.Model.Version,
		DenseParameters: s.Model.DenseParameters,
	}
	return &resp, nil
}

// PullEmbeddingTable pulls sparse parameter from server
func (s *Server) PullEmbeddingTable(ctx context.Context, in *proto.PullEmbeddingTableRequest) (*proto.Tensor, error) {
	return s.Model.EmbeddingTables[in.Name].GetEmbeddingVectors(in.Ids), nil
}

// PushGradients push gradients to server
func (s *Server) PushGradients(ctx context.Context, in *proto.Model) (*proto.PushGradientResponse, error) {
	err := s.Optimizer.ApplyGradients(in, s.Model)
	var resp = proto.PushGradientResponse{
		Accepted: true,
		Version:  s.Model.Version,
	}
	return &resp, err
}

// PushModel push Model to server
func (s *Server) PushModel(ctx context.Context, in *proto.Model) (*empty.Empty, error) {
	s.Model.InitFromModelPB(in)
	s.Model.InitStatus = true
	return &empty.Empty{}, nil
}

// Run creates a grpc server and starts the serving. Set serverDone when finishes.
func (s *Server) Run(address string, serverDone chan bool) *grpc.Server {
	lis, err := net.Listen("tcp", address)
	if err != nil {
		log.Fatalf("failed to start PS: %v", err)
	}
	// TODO: set maxReceiveMessageSize (default is 4M, too small for elasticdl), maxConcurrentStreams
	grpcServer := grpc.NewServer()
	proto.RegisterPserverServer(grpcServer, s)
	go startServe(grpcServer, lis, serverDone)
	return grpcServer
}

func startServe(server *grpc.Server, lis net.Listener, serverDone chan bool) {
	err := server.Serve(lis)
	if err != nil {
		log.Fatalf("GRPC failed to serve: %v", err)
	}
	serverDone <- true
}
