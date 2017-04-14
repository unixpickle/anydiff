package anyfwd

import (
	"testing"

	"github.com/unixpickle/anyvec"
)

func TestCreatorConcat(t *testing.T) {
	tester := NewTester(t)
	tester.TestVecFunc(15, func(in anyvec.Vector) anyvec.Vector {
		in1 := in.Slice(0, 4)
		in2 := in.Slice(4, 8)
		in3 := in.Slice(8, 15)
		return in.Creator().Concat(in2, in3, in1)
	})
}
