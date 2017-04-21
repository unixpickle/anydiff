package anyfwd

import (
	"testing"

	"github.com/unixpickle/anyvec"
)

func TestNumericOps(t *testing.T) {
	tester := NewTester(t)
	tester.TestVecFunc(4, func(in anyvec.Vector) anyvec.Vector {
		in.Slice(0, 1).AddScalar(in.Creator().MakeNumeric(10))
		n1 := tester.GetComponent(in, 0)
		n2 := tester.GetComponent(in, 1)
		n3 := tester.GetComponent(in, 2)
		n4 := tester.GetComponent(in, 3)
		ops := in.Creator().NumOps()

		// Evaluate n1 + n2*(n4-n3)/n1.
		ans := ops.Add(n1, ops.Div(ops.Mul(n2, ops.Sub(n4, n3)), n1))

		res := in.Creator().MakeVector(1)
		res.AddScalar(ans)
		return res
	})
}
