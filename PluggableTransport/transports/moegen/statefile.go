/*
 * Copyright (c) 2014, Yawning Angel <yawning at schwanenlied dot me>
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

package moegen

import (
	"PluggableTransport/common/log"
	"encoding/base64"
	"encoding/json"
	"fmt"
	pt "git.torproject.org/pluggable-transports/goptlib.git"
	"io/ioutil"
	"os"
	"path"
	"strings"

	"PluggableTransport/common/csrand"
	"PluggableTransport/common/drbg"
	"PluggableTransport/common/ntor"
)

const (
	stateFile  = "moegen_state.json"
	bridgeFile = "moegen_bridgeline.txt"

	certSuffix = "=="
	certLength = ntor.NodeIDLength + ntor.PublicKeyLength
)

type jsonServerState struct {
	NodeID             string `json:"node-id"`
	PrivateKey         string `json:"private-key"`
	PublicKey          string `json:"public-key"`
	DrbgSeed           string `json:"drbg-seed"`
	PerturbationString string
}

type moegenServerCert struct {
	raw []byte
}

func (cert *moegenServerCert) String() string {
	return strings.TrimSuffix(base64.StdEncoding.EncodeToString(cert.raw), certSuffix)
}

func (cert *moegenServerCert) unpack() (*ntor.NodeID, *ntor.PublicKey) {
	if len(cert.raw) != certLength {
		panic(fmt.Sprintf("cert length %d is invalid", len(cert.raw)))
	}

	nodeID, _ := ntor.NewNodeID(cert.raw[:ntor.NodeIDLength])
	pubKey, _ := ntor.NewPublicKey(cert.raw[ntor.NodeIDLength:])

	return nodeID, pubKey
}

func serverCertFromString(encoded string) (*moegenServerCert, error) {
	decoded, err := base64.StdEncoding.DecodeString(encoded + certSuffix)
	if err != nil {
		return nil, fmt.Errorf("failed to decode cert: %s", err)
	}

	if len(decoded) != certLength {
		return nil, fmt.Errorf("cert length %d is invalid", len(decoded))
	}

	return &moegenServerCert{raw: decoded}, nil
}

func serverCertFromState(st *moegenServerState) *moegenServerCert {
	cert := new(moegenServerCert)
	cert.raw = append(st.nodeID.Bytes()[:], st.identityKey.Public().Bytes()[:]...)
	return cert
}

type moegenServerState struct {
	nodeID             *ntor.NodeID
	identityKey        *ntor.Keypair
	drbgSeed           *drbg.Seed
	PerturbationString string

	cert *moegenServerCert
}

func (st *moegenServerState) clientString() string {
	return fmt.Sprintf("%s=%s", certArg, st.cert)
}

func serverStateFromArgs(stateDir string, args *pt.Args) (*moegenServerState, error) {
	var js jsonServerState
	var nodeIDOk, privKeyOk, seedOk bool

	log.Noticef("-----------[statefile.serverStateFromArgs()] Begin get state from args")
	fmt.Println(fmt.Errorf("-----------[statefile.serverStateFromArgs()] Begin get state from args"))
	js.NodeID, nodeIDOk = args.Get(nodeIDArg)
	js.PrivateKey, privKeyOk = args.Get(privateKeyArg)
	js.DrbgSeed, seedOk = args.Get(seedArg)
	pertPathStr, pertOk := args.Get(perturbationsArg) // if pertOk, then replace json.PerturbationString (by jsonServerStateFromFile)

	// Either a private key, node id, and seed are ALL specified, or
	// they should be loaded from the state file.
	if !privKeyOk && !nodeIDOk && !seedOk { // do not && !pertOk, cause jsonServerStateFromFile will create new privateKey
		log.Noticef("-----------[statefile.serverStateFromArgs()] Create new json state")
		fmt.Println(fmt.Errorf("-----------[statefile.serverStateFromArgs()] Create new json state"))
		if err := jsonServerStateFromFile(stateDir, &js); err != nil {
			return nil, err
		}
	} else if !privKeyOk {
		return nil, fmt.Errorf("-----------[statefile.serverStateFromArgs()]missing argument '%s'", privateKeyArg)
	} else if !nodeIDOk {
		return nil, fmt.Errorf("-----------[statefile.serverStateFromArgs()]missing argument '%s'", nodeIDArg)
	} else if !seedOk {
		return nil, fmt.Errorf("-----------[statefile.serverStateFromArgs()]missing argument '%s'", seedArg)
	} else if !pertOk {
		return nil, fmt.Errorf("-----------[statefile.serverStateFromArgs()] missing argument '%s'", perturbationsArg)
	}
	if pertOk {
		// if torrc-server has perturbations, then replace old perturbations in state file.
		log.Noticef("-----------[statefile.serverStateFromArgs()] Overriding perturbations from args: %s", pertPathStr)
		fmt.Println(fmt.Errorf("-----------[statefile.serverStateFromArgs()] Overriding perturbations from args: %s", pertPathStr))
		js.PerturbationString = pertPathStr
	} else {
		// read old perturbations
		log.Noticef("-----------[statefile.serverStateFromArgs()] Keeping perturbations from JSON: %s", js.PerturbationString)
		fmt.Println(fmt.Errorf("-----------[statefile.serverStateFromArgs()] Keeping perturbations from JSON: %s", js.PerturbationString))
	}

	log.Noticef("-----------[statefile.serverStateFromArgs()] Already got state from args")
	fmt.Println(fmt.Errorf("-----------[statefile.serverStateFromArgs()] Already got state from args"))
	return serverStateFromJSONServerState(stateDir, &js)
}

func serverStateFromJSONServerState(stateDir string, js *jsonServerState) (*moegenServerState, error) {
	var err error

	log.Noticef("-----------[statefile.serverStateFromJSONServerState()] Begin create ServerState")
	fmt.Println(fmt.Errorf("-----------[statefile.serverStateFromJSONServerState()] Begin create ServerState"))
	st := new(moegenServerState)
	st.PerturbationString = js.PerturbationString // record PerturbationString (str list)
	if st.nodeID, err = ntor.NodeIDFromHex(js.NodeID); err != nil {
		return nil, err
	}
	if st.identityKey, err = ntor.KeypairFromHex(js.PrivateKey); err != nil {
		return nil, err
	}
	if st.drbgSeed, err = drbg.SeedFromHex(js.DrbgSeed); err != nil {
		return nil, err
	}
	st.cert = serverCertFromState(st)

	// Generate a human readable summary of the configured endpoint.
	if err = newBridgeFile(stateDir, st); err != nil {
		return nil, err
	}

	log.Noticef("-----------[statefile.serverStateFromJSONServerState()] Already created ServerState")
	fmt.Println(fmt.Errorf("-----------[statefile.serverStateFromJSONServerState()] Already created ServerState"))
	// Write back the possibly updated server state.
	return st, writeJSONServerState(stateDir, js)
}

func jsonServerStateFromFile(stateDir string, js *jsonServerState) error {
	fPath := path.Join(stateDir, stateFile)
	f, err := ioutil.ReadFile(fPath)
	if err != nil {
		if os.IsNotExist(err) {
			if err = newJSONServerState(stateDir, js); err == nil {
				return nil
			}
		}
		return err
	}

	if err := json.Unmarshal(f, js); err != nil {
		return fmt.Errorf("failed to load statefile '%s': %s", fPath, err)
	}

	return nil
}

func newJSONServerState(stateDir string, js *jsonServerState) (err error) {
	// Generate everything a server needs, using the cryptographic PRNG.
	var st moegenServerState
	rawID := make([]byte, ntor.NodeIDLength)
	if err = csrand.Bytes(rawID); err != nil {
		return
	}
	if st.nodeID, err = ntor.NewNodeID(rawID); err != nil {
		return
	}
	if st.identityKey, err = ntor.NewKeypair(false); err != nil {
		return
	}
	if st.drbgSeed, err = drbg.NewSeed(); err != nil {
		return
	}

	// Encode it into JSON format and write the state file.
	js.NodeID = st.nodeID.Hex()
	js.PrivateKey = st.identityKey.Private().Hex()
	js.PublicKey = st.identityKey.Public().Hex()
	js.DrbgSeed = st.drbgSeed.Hex()
	js.PerturbationString = "[]" // initial is empty

	return writeJSONServerState(stateDir, js)
}

func writeJSONServerState(stateDir string, js *jsonServerState) error {
	var err error
	var encoded []byte
	if encoded, err = json.Marshal(js); err != nil {
		return err
	}
	if err = ioutil.WriteFile(path.Join(stateDir, stateFile), encoded, 0600); err != nil {
		log.Noticef("-----------[statefile.writeJSONServerState()] Write new json state at [%s] filed, due to: %s", stateDir, err)
		fmt.Println(fmt.Errorf("-----------[statefile.writeJSONServerState()] Write new json state at [%s] filed, due to: %s", stateDir, err))
		return err
	}

	return nil
}

func newBridgeFile(stateDir string, st *moegenServerState) error {
	const prefix = "# moegen torrc client bridge line\n" +
		"#\n" +
		"# This file is an automatically generated bridge line based on\n" +
		"# the current moegenproxy configuration.  EDITING IT WILL HAVE\n" +
		"# NO EFFECT.\n" +
		"#\n" +
		"# Before distributing this Bridge, edit the placeholder fields\n" +
		"# to contain the actual values:\n" +
		"#  <IP ADDRESS>  - The public IP address of your moegen bridge.\n" +
		"#  <PORT>        - The TCP/IP port of your moegen bridge.\n" +
		"#  <FINGERPRINT> - The bridge's fingerprint.\n\n"

	bridgeLine := fmt.Sprintf("Bridge moegen <IP ADDRESS>:<PORT> <FINGERPRINT> %s\n",
		st.clientString())

	tmp := []byte(prefix + bridgeLine)
	if err := ioutil.WriteFile(path.Join(stateDir, bridgeFile), tmp, 0600); err != nil {
		log.Noticef("-----------[statefile.newBridgeFile()] Write new BridgeFile at [%s] filed, due to: %s", stateDir, err)
		fmt.Println(fmt.Errorf("-----------[statefile.newBridgeFile()] Write new BridgeFile at [%s] filed, due to: %s", stateDir, err))
		return err
	}

	return nil
}
