#!/bin/sh

{
        set -e

        install_dir="/usr/local/bin"
        install_path="$install_dir/beam"
        latest_release_uri="https://github.com/slai-labs/get-beam/releases"

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
        # 
        latest_release_tag=$(curl -sL -o /dev/null -w %{url_effective} $latest_release_uri/latest | rev | cut -d/ -f1 | rev)

        echo "Latest release: $latest_release_tag"
        latest_beam_binary_uri="$latest_release_uri/download/$latest_release_tag/beam"

        download_path=$(mktemp)
        echo "Downloading latest beam binary $latest_beam_binary_uri\n"
        curl -L $latest_beam_binary_uri -o $download_path

        chmod +x $download_path

        $sh_c "mv -f $download_path $install_path"

        printf "\033[32m\nSuccessfully installed latest version of Beam: $latest_release_tag\n\033[0m"
}