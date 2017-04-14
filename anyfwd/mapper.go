package anyfwd

import "github.com/unixpickle/anyvec"

type mapper struct {
	creator *Creator
}

func (m *mapper) Creator() anyvec.Creator {
	return m.creator
}

func (m *mapper) InSize() int {
	panic("nyi")
}

func (m *mapper) OutSize() int {
	panic("nyi")
}

func (m *mapper) Map(in, out anyvec.Vector) {
	panic("nyi")
}

func (m *mapper) MapTranspose(in, out anyvec.Vector) {
	panic("nyi")
}
