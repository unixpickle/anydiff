package anyfwd

import (
	"math/rand"

	"github.com/unixpickle/anyvec"
)

// Rand sets the vector to random values.
// The gradients will all be set to zero.
func (v *Vector) Rand(p anyvec.ProbDist, r *rand.Rand) {
	anyvec.Rand(v.Values, p, r)
	v.clearJacobian()
}
