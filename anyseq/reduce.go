package anyseq

import "github.com/unixpickle/anyvec"

// ReduceBatch eliminates sequences in b to get a new
// batch with the requested present map.
//
// It is invalid for present[i] to be true when
// b.Present[i] is false.
func ReduceBatch(b *Batch, present []bool) *Batch {
	n := b.NumPresent()
	inc := b.Packed.Len() / n

	var chunks []anyvec.Vector
	var chunkStart, chunkSize int
	for i, pres := range present {
		if pres {
			if !b.Present[i] {
				panic("cannot re-add sequences")
			}
			chunkSize += inc
		} else if b.Present[i] {
			if chunkSize > 0 {
				chunks = append(chunks, b.Packed.Slice(chunkStart, chunkStart+chunkSize))
				chunkStart += chunkSize
				chunkSize = 0
			}
			chunkStart += inc
		}
	}
	if chunkSize > 0 {
		chunks = append(chunks, b.Packed.Slice(chunkStart, chunkStart+chunkSize))
	}

	return &Batch{
		Packed:  b.Packed.Creator().Concat(chunks...),
		Present: present,
	}
}

// ExpandBatch reverses the process of ReduceBatch by
// inserting zero entries in the batch to get a desired
// present map.
//
// It is invalid for present[i] to be false when
// b.Present[i] is true.
func ExpandBatch(b *Batch, present []bool) *Batch {
	n := b.NumPresent()
	inc := b.Packed.Len() / n
	filler := b.Packed.Creator().MakeVector(inc)

	var chunks []anyvec.Vector
	var chunkStart, chunkSize int

	for i, pres := range present {
		if b.Present[i] {
			if !pres {
				panic("argument to Expand must be a superset")
			}
			chunkSize += inc
		} else if pres {
			if chunkSize > 0 {
				chunks = append(chunks, b.Packed.Slice(chunkStart, chunkStart+chunkSize))
				chunkStart += chunkSize
				chunkSize = 0
			}
			chunks = append(chunks, filler)
		}
	}
	if chunkSize > 0 {
		chunks = append(chunks, b.Packed.Slice(chunkStart, chunkSize+chunkStart))
	}

	return &Batch{
		Packed:  b.Packed.Creator().Concat(chunks...),
		Present: present,
	}
}
