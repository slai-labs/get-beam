#!/bin/sh

{ # Prevent execution if this script was only partially downloaded

        set -e

        install_dir='/usr/local/bin'
        install_path='/usr/local/bin/beam'
        OS=$(uname | tr '[:upper:]' '[:lower:]')
        ARCH=$(uname -m | tr '[:upper:]' '[:lower:]')
        cmd_exists() {
                command -v "$@" >/dev/null 2>&1
        }

	if [ ! -d "$install_dir" ]; then
                echo "Install directory $install_dir doesn't exist. Creating now."
                if ! mkdir -p -- "$install_dir"; then
                        echo "Failed to create the install directory: $install_dir"
                        exit 1
                fi
                echo "Install directory created."
        fi

        bin_file=

        case "$OS" in
        darwin)
                case "$ARCH" in
                x86_64)
                        bin_file=beam-Darwin-x86_64
                        ;;
                arm64)
                        bin_file=beam-Darwin-arm64
                        ;;
                *)
                        printf '\033[31m> The architecture (%s) is not supported by this installation script.\n\033[0m' "$ARCH"
                        exit 1
                        ;;
                esac
                ;;
        linux)
                case "$ARCH" in
                x86_64)
                        bin_file=beam-Linux-x86_64
                        ;;
                amd64)
                        bin_file=beam-Linux-x86_64
                        ;;
                armv8*)
                        bin_file=beam-Linux-arm64
                        ;;
                aarch64)
                        bin_file=beam-Linux-arm64
                        ;;
                *)
                        printf '\033[31m> The architecture (%s) is not supported by this installation script.\n\033[0m' "$ARCH"
                        exit 1
                        ;;
                esac
                ;;
        *)
                printf '\033[31m> The OS (%s) is not supported by this installation script.\n\033[0m' "$OS"
                exit 1
                ;;
        esac

        sh_c='sh -c'
        if [ ! -w "$install_dir" ]; then
                # use sudo if $USER doesn't have write access to the path
                if [ "$USER" != 'root' ]; then
                        if cmd_exists sudo; then
                                sh_c='sudo -E sh -c'
                        elif cmd_exists su; then
                                sh_c='su -c'
                        else
                                echo 'This script requires to run commands as sudo. We are unable to find either "sudo" or "su".'
                                exit 1
                        fi
                fi
        fi

        echo "Getting latest release"
        latest_release_uri="https://github.com/slai-labs/get-beam/releases"
        latest_release_tag=$(curl -sL -o /dev/null -w %{url_effective} $latest_release_uri/latest | rev | cut -d/ -f1 | rev)

        echo "Latest release: $latest_release_tag"
        latest_beam_binary_uri="$latest_release_uri/download/$latest_release_tag/$bin_file"

        download_path=$(mktemp)
        echo "Downloading latest beam binary $latest_beam_binary_uri\n"
        curl -fSL $latest_beam_binary_uri -o $download_path
        chmod +x $download_path

        printf '> Installing %s\n' "$install_path"
        $sh_c "mv -f $download_path $install_path"

        printf '\033[32m> Beam successfully installed!\n\033[0m'

} # End of wrapping
