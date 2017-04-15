package anyfwd

import (
	"testing"

	"github.com/unixpickle/anyvec"
)

func TestComplement(t *testing.T) {
	testUnaryOp(t, anyvec.Complement)
}
