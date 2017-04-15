package anyfwd

import (
	"testing"

	"github.com/unixpickle/anyvec"
)

func TestAddChunks(t *testing.T) {
	tester := NewTester(t)
	tester.TestVecFunc(6+3, func(in anyvec.Vector) anyvec.Vector {
		v1 := in.Slice(0, 6)
		v2 := in.Slice(6, 9)
		anyvec.AddChunks(v1, v2)
		return in
	})
}

func TestScaleChunks(t *testing.T) {
	tester := NewTester(t)
	tester.TestVecFunc(6+3, func(in anyvec.Vector) anyvec.Vector {
		v1 := in.Slice(0, 6)
		v2 := in.Slice(6, 9)
		anyvec.ScaleChunks(v1, v2)
		return in
	})
}

func TestAddRepeated(t *testing.T) {
	tester := NewTester(t)
	tester.TestVecFunc(15+7, func(in anyvec.Vector) anyvec.Vector {
		v1 := in.Slice(0, 15)
		v2 := in.Slice(15, 15+7)
		anyvec.AddRepeated(v1, v2)
		return in
	})
}

func TestScaleRepeated(t *testing.T) {
	tester := NewTester(t)
	tester.TestVecFunc(15+7, func(in anyvec.Vector) anyvec.Vector {
		v1 := in.Slice(0, 15)
		v2 := in.Slice(15, 15+7)
		anyvec.ScaleRepeated(v1, v2)
		return in
	})
}

func TestSumRows(t *testing.T) {
	tester := NewTester(t)
	tester.TestVecFunc(18, func(in anyvec.Vector) anyvec.Vector {
		return anyvec.SumRows(in, 6)
	})
}

func TestSumCols(t *testing.T) {
	tester := NewTester(t)
	tester.TestVecFunc(18, func(in anyvec.Vector) anyvec.Vector {
		return anyvec.SumRows(in, 3)
	})
}
