package anydiff

import (
	"fmt"

	"github.com/unixpickle/anyvec/anyvec32"
)

func ExamplePool() {
	v := NewVar(anyvec32.MakeVectorData([]float32{1, 2, 3}))

	// Compute v^2 * (v^2 - 1) while only back-propagating
	// through v^2 once.
	out := Pool(Mul(v, v), func(v2 Res) Res {
		bias := NewConst(anyvec32.MakeVectorData([]float32{-1}))
		return Mul(v2, AddRepeated(v2, bias))
	})

	fmt.Println(out.Output().Data())

	// Output: [0 12 72]
}
