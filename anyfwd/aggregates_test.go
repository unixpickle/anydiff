package anyfwd

import (
	"testing"

	"github.com/unixpickle/anyvec"
)

func TestSum(t *testing.T) {
	testAggregate(t, anyvec.Sum)
}

func TestMax(t *testing.T) {
	// TODO: ensure that this test is reliable in the
	// face of forward auto-diff.
	testAggregate(t, anyvec.Max)
}

func TestAbsSum(t *testing.T) {
	testAggregate(t, anyvec.AbsSum)
}

func TestAbsMax(t *testing.T) {
	// TODO: ensure numerical stability (see TestMax).
	testAggregate(t, anyvec.AbsMax)
}

func TestNorm(t *testing.T) {
	testAggregate(t, anyvec.Norm)
}

func testAggregate(t *testing.T, f func(in anyvec.Vector) anyvec.Numeric) {
	tester := NewTester(t)
	tester.TestVecFunc(15, func(in anyvec.Vector) anyvec.Vector {
		resVec := in.Creator().MakeVector(1)
		resVec.AddScalar(f(in))
		return resVec
	})
}
